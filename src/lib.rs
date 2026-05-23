//! Set-associative cache with CLOCK eviction, SIMD tag matching, and custom allocator support.

#![allow(clippy::identity_op)]

use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ptr::NonNull;

pub mod alloc;
use alloc::{Allocator, Global};

/// `#[cold]` marks the function as rarely called, making the opposite branch
/// the predicted path. Used to emulate `likely`/`unlikely` on stable.
#[inline(always)]
#[cold]
fn cold_path() {}

/// Hint: condition is likely true.
#[inline(always)]
#[allow(unused)]
fn likely(b: bool) -> bool {
    if b {
        true
    } else {
        cold_path();
        false
    }
}

/// Hint: condition is likely false.
#[inline(always)]
fn unlikely(b: bool) -> bool {
    if b {
        cold_path();
        true
    } else {
        false
    }
}

/// Strategy for extracting a key from a stored value.
///
/// Allows storing only values when the key is embedded in the value.
/// For simple `(K, V)` pairs, use [`PairExtract`].
pub trait KeyExtract {
    /// The key type used for lookups.
    type Key: Hash + Eq;
    /// The value type stored in the cache.
    type Value;

    /// Extract a key reference from a stored value.
    fn extract(value: &Self::Value) -> &Self::Key;
}

/// Standard `(K, V)` extraction — key is the first element.
pub struct PairExtract<K, V>(PhantomData<fn() -> (K, V)>);

impl<K: Hash + Eq, V> KeyExtract for PairExtract<K, V> {
    type Key = K;
    type Value = (K, V);

    #[inline]
    fn extract(value: &(K, V)) -> &K {
        &value.0
    }
}

/// Key equivalence trait.
///
/// This trait defines the function used to compare the input value with the
/// cache keys during a lookup operation such as [`SetAssociativeCache::get`].
/// It is provided with a blanket implementation based on the
/// [`Borrow`](core::borrow::Borrow) trait.
///
/// # Correctness
///
/// Equivalent values must hash to the same value.
pub trait Equivalent<K: ?Sized> {
    /// Checks if this value is equivalent to the given key.
    fn equivalent(&self, key: &K) -> bool;
}

impl<Q: ?Sized, K: ?Sized> Equivalent<K> for Q
where
    Q: Eq,
    K: core::borrow::Borrow<Q>,
{
    #[inline(always)]
    fn equivalent(&self, key: &K) -> bool {
        self == key.borrow()
    }
}

// =============================================================================
// CacheLayout
// =============================================================================

/// Compile-time cache geometry parameters.
pub trait CacheLayout {
    /// Number of ways per set (2, 4, or 16).
    const WAYS: u64;
    /// Tag bits per entry (8 or 16).
    const TAG_BITS: u64;
    /// CLOCK counter bits per entry (1, 2, or 4).
    const CLOCK_BITS: u64;
    /// Hardware cache line size in bytes (power of two).
    const CACHE_LINE_SIZE: u64;
}

/// Default cache layout: 16 ways, 8-bit tags, 2-bit CLOCK counters, 64-byte cache lines.
pub struct DefaultLayout;

impl CacheLayout for DefaultLayout {
    const WAYS: u64 = 16;
    const TAG_BITS: u64 = 8;
    const CLOCK_BITS: u64 = 2;
    const CACHE_LINE_SIZE: u64 = 64;
}

/// Whether an upsert updated an existing entry or inserted a new one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateOrInsert {
    /// Existing entry was updated.
    Update,
    /// New entry was inserted.
    Insert,
}

/// Result of an [`SetAssociativeCache::upsert`] operation.
#[derive(Debug)]
pub struct UpsertResult<V> {
    /// Slot index where the value was placed.
    pub index: usize,
    /// Whether the operation was an update or insert.
    pub updated: UpdateOrInsert,
    /// The evicted value, if any.
    pub evicted: Option<V>,
}

/// Cache hit/miss counters.
#[derive(Debug, Default, Clone)]
pub struct Metrics {
    /// Number of cache hits.
    pub hits: u64,
    /// Number of cache misses.
    pub misses: u64,
    /// Current number of live entries.
    pub value_count: u64,
}

#[inline(always)]
const fn log2(x: u64) -> u64 {
    assert!(x.is_power_of_two() && x > 0);
    x.trailing_zeros() as u64
}

/// Fast alternative to modulo reduction.
/// See <https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/>
#[inline(always)]
pub fn fastrange(word: u64, p: u64) -> u64 {
    ((word as u128).wrapping_mul(p as u128) >> 64) as u64
}

/// Integer division rounding up.
#[inline]
pub fn div_ceil(numerator: u64, denominator: u64) -> u64 {
    assert!(denominator > 0);
    if numerator == 0 {
        return 0;
    }
    numerator.div_ceil(denominator)
}

// =============================================================================
// AlignedBuf — raw aligned allocation via Allocator
// =============================================================================

struct AlignedBuf<T> {
    ptr: NonNull<T>,
    len: usize,
    layout: std::alloc::Layout,
}

impl<T> AlignedBuf<T> {
    /// Allocate `len` elements of type `T` with at least `align` byte alignment.
    /// All bytes are zeroed.
    fn alloc_zeroed(len: usize, align: usize, alloc: &impl Allocator) -> Self {
        if len == 0 || std::mem::size_of::<T>() == 0 {
            return Self {
                ptr: NonNull::dangling(),
                len,
                layout: std::alloc::Layout::from_size_align(
                    0,
                    align.max(std::mem::align_of::<T>()),
                )
                .unwrap(),
            };
        }
        let size = len * std::mem::size_of::<T>();
        let align = align.max(std::mem::align_of::<T>());
        let layout = std::alloc::Layout::from_size_align(size, align).unwrap();
        let slice = alloc::do_alloc(alloc, layout).expect("allocation failed");
        let ptr = slice.as_ptr().cast::<u8>();
        // SAFETY: ptr is valid for `size` bytes, freshly allocated.
        unsafe { std::ptr::write_bytes(ptr, 0, size) };
        Self {
            // SAFETY: ptr is non-null (from successful allocation) and properly aligned.
            ptr: unsafe { NonNull::new_unchecked(ptr.cast::<T>()) },
            len,
            layout,
        }
    }
    #[inline(always)]
    fn as_slice(&self) -> &[T] {
        if unlikely(self.len == 0) {
            return &[];
        }
        // SAFETY: ptr is valid for `len` elements, allocated and zeroed in alloc_zeroed.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [T] {
        if unlikely(self.len == 0) {
            return &mut [];
        }
        // SAFETY: ptr is valid for `len` elements, we have exclusive access via &mut self.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Deallocate using the given allocator. Must use the same allocator that allocated.
    ///
    /// # SAFETY:
    /// Must only be called once, with the same allocator used for allocation.
    unsafe fn dealloc(&self, alloc: &impl Allocator) {
        if unlikely(self.layout.size() == 0) {
            return;
        }
        // SAFETY: caller guarantees this is the matching allocator and single dealloc call.
        unsafe {
            alloc.deallocate(
                NonNull::new_unchecked(self.ptr.as_ptr().cast::<u8>()),
                self.layout,
            );
        }
    }
    #[inline(always)]
    fn fill(&mut self, val: T)
    where
        T: Copy,
    {
        self.as_mut_slice().fill(val);
    }
}

/// Bit-packed array of small unsigned integers (1, 2, or 4 bits) stored in u64 words.
#[derive(Debug)]
pub struct PackedArray {
    uint_bits: u32,
    words: PackedWords,
}

#[derive(Debug)]
enum PackedWords {
    Vec(Vec<u64>),
    Buf {
        ptr: NonNull<u64>,
        len: usize,
        layout: std::alloc::Layout,
    },
}

impl PackedWords {
    #[inline]
    fn as_slice(&self) -> &[u64] {
        match self {
            PackedWords::Vec(v) => v,
            PackedWords::Buf { ptr, len, .. } => {
                if *len == 0 {
                    return &[];
                }
                // SAFETY: ptr is valid for `len` u64 elements, from AlignedBuf allocation.
                unsafe { std::slice::from_raw_parts(ptr.as_ptr(), *len) }
            }
        }
    }
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [u64] {
        match self {
            PackedWords::Vec(v) => v,
            PackedWords::Buf { ptr, len, .. } => {
                if *len == 0 {
                    return &mut [];
                }
                // SAFETY: ptr is valid for `len` u64 elements, we have exclusive access.
                unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr(), *len) }
            }
        }
    }
}

impl PackedArray {
    /// Create a new packed array using the global allocator (Vec-backed).
    pub fn new(uint_bits: u32, count: u64) -> Self {
        assert!(uint_bits == 1 || uint_bits == 2 || uint_bits == 4);
        let total_bits = count * uint_bits as u64;
        let num_words = div_ceil(total_bits, 64);
        Self {
            uint_bits,
            words: PackedWords::Vec(vec![0u64; num_words as usize]),
        }
    }

    fn new_aligned(uint_bits: u32, count: u64, align: usize, alloc: &impl Allocator) -> Self {
        assert!(uint_bits == 1 || uint_bits == 2 || uint_bits == 4);
        let total_bits = count * uint_bits as u64;
        let num_words = div_ceil(total_bits, 64) as usize;
        let buf = AlignedBuf::<u64>::alloc_zeroed(num_words, align, alloc);
        Self {
            uint_bits,
            words: PackedWords::Buf {
                ptr: buf.ptr,
                len: buf.len,
                layout: buf.layout,
            },
        }
    }

    /// Get the value at `index`.
    #[inline]
    pub fn get(&self, index: u64) -> u64 {
        let words = self.words.as_slice();
        let uint_bits = self.uint_bits;
        let uints_per_word = 64 / uint_bits;
        let word_idx = (index / uints_per_word as u64) as usize;
        let bit_offset = (index % uints_per_word as u64) * uint_bits as u64;
        let mask = (1u64 << uint_bits) - 1;
        (words[word_idx] >> bit_offset) & mask
    }

    /// Set the value at `index`.
    #[inline]
    pub fn set(&mut self, index: u64, value: u64) {
        let words = self.words.as_mut_slice();
        let uint_bits = self.uint_bits;
        let uints_per_word = 64 / uint_bits;
        let word_idx = (index / uints_per_word as u64) as usize;
        let bit_offset = (index % uints_per_word as u64) * uint_bits as u64;
        let mask = (1u64 << uint_bits) - 1;
        words[word_idx] &= !(mask << bit_offset);
        words[word_idx] |= (value & mask) << bit_offset;
    }

    /// Zero all entries.
    pub fn clear(&mut self) {
        self.words.as_mut_slice().fill(0);
    }

    /// View the underlying u64 words.
    pub fn words(&self) -> &[u64] {
        self.words.as_slice()
    }

    /// Deallocate the buffer variant. No-op for Vec variant.
    unsafe fn dealloc(&self, alloc: &impl Allocator) {
        if let PackedWords::Buf { ptr, layout, .. } = &self.words
            && layout.size() > 0
        {
            // SAFETY: caller guarantees matching allocator and single dealloc.
            unsafe {
                alloc.deallocate(NonNull::new_unchecked(ptr.as_ptr().cast::<u8>()), *layout);
            }
        }
    }
}

mod simd {
    /// Compare `ways` u8 tags against `needle`, return bitmask of matching positions.
    /// Runtime SIMD on x86_64 (SSE2), scalar fallback everywhere else.
    #[inline]
    pub(crate) fn search_tags(tags: &[u8], needle: u8, ways: u64) -> u64 {
        #[cfg(target_arch = "x86_64")]
        {
            if ways == 16 {
                if is_x86_feature_detected!("avx512bw") {
                    // SAFETY: tags slice has 16 elements and is 64-byte aligned (from AlignedBuf).
                    return unsafe { search_tags_16_avx512(tags, needle) };
                }
                if is_x86_feature_detected!("sse2") {
                    // SAFETY: tags slice has 16 elements and is 16-byte aligned (from AlignedBuf).
                    return unsafe { search_tags_16_sse2(tags, needle) };
                }
            }
        }
        search_tags_scalar(tags, needle, ways)
    }

    /// Compare `ways` u16 tags against `needle`, return bitmask of matching positions.
    /// Runtime SIMD on x86_64 (AVX2 -> SSE2), scalar fallback everywhere else.
    #[inline]
    pub(crate) fn search_tags_u16(tags: &[u16], needle: u16, ways: u64) -> u64 {
        #[cfg(target_arch = "x86_64")]
        {
            if ways == 16 {
                if is_x86_feature_detected!("avx512bw") {
                    // SAFETY: tags slice has 16 elements and is 64-byte aligned (from AlignedBuf).
                    return unsafe { search_tags_u16_16_avx512(tags, needle) };
                }
                if is_x86_feature_detected!("avx2") {
                    // SAFETY: tags slice has 16 elements and is 32-byte aligned (from AlignedBuf).
                    return unsafe { search_tags_u16_16_avx2(tags, needle) };
                }
                if is_x86_feature_detected!("sse2") {
                    // SAFETY: tags slice has 16 elements and is 16-byte aligned (from AlignedBuf).
                    return unsafe { search_tags_u16_16_sse2(tags, needle) };
                }
            }
        }
        search_tags_u16_scalar(tags, needle, ways)
    }

    // ---- x86_64 AVX-512: 16 × u8 (single 128-bit compare with mask result) ----

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512bw,avx512vl")]
    unsafe fn search_tags_16_avx512(tags: &[u8], needle: u8) -> u64 {
        use std::arch::x86_64::*;
        // SAFETY: caller guarantees tags is 16-byte aligned and has >= 16 elements.
        unsafe {
            let data = _mm_load_si128(tags.as_ptr().cast::<__m128i>());
            let splat = _mm_set1_epi8(needle as i8);
            let mask = _mm_cmpeq_epi8_mask(data, splat);
            (mask & 0xFFFF) as u64
        }
    }

    // ---- x86_64 AVX-512: 16 × u16 (single 256-bit compare with mask result) ----

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512bw,avx512vl")]
    unsafe fn search_tags_u16_16_avx512(tags: &[u16], needle: u16) -> u64 {
        use std::arch::x86_64::*;
        // SAFETY: caller guarantees tags is 32-byte aligned and has >= 16 elements.
        unsafe {
            let data = _mm256_load_si256(tags.as_ptr().cast::<__m256i>());
            let splat = _mm256_set1_epi16(needle as i16);
            let mask = _mm256_cmpeq_epi16_mask(data, splat);
            (mask & 0xFFFF) as u64
        }
    }

    // ---- x86_64 SSE2: 16 × u8 ----

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2")]
    unsafe fn search_tags_16_sse2(tags: &[u8], needle: u8) -> u64 {
        use std::arch::x86_64::*;
        // SAFETY: caller guarantees tags is 16-byte aligned and has >= 16 elements.
        unsafe {
            let data = _mm_load_si128(tags.as_ptr().cast::<__m128i>());
            let splat = _mm_set1_epi8(needle as i8);
            let cmp = _mm_cmpeq_epi8(data, splat);
            let mask = _mm_movemask_epi8(cmp) as u32;
            (mask & 0xFFFF) as u64
        }
    }

    // ---- x86_64 SSE2: 16 × u16 (two 128-bit passes) ----

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2")]
    unsafe fn search_tags_u16_16_sse2(tags: &[u16], needle: u16) -> u64 {
        use std::arch::x86_64::*;
        // SAFETY: caller guarantees tags is 32-byte aligned and has >= 16 elements.
        // Each half (8 × u16 = 16 bytes) is 16-byte aligned.
        unsafe {
            let splat = _mm_set1_epi16(needle as i16);

            let lo = _mm_load_si128(tags.as_ptr().cast::<__m128i>());
            let cmp_lo = _mm_cmpeq_epi16(lo, splat);
            let packed_lo = _mm_packs_epi16(cmp_lo, _mm_setzero_si128());
            let mask_lo = _mm_movemask_epi8(packed_lo) as u32 & 0xFF;

            let hi = _mm_load_si128(tags.as_ptr().add(8).cast::<__m128i>());
            let cmp_hi = _mm_cmpeq_epi16(hi, splat);
            let packed_hi = _mm_packs_epi16(cmp_hi, _mm_setzero_si128());
            let mask_hi = _mm_movemask_epi8(packed_hi) as u32 & 0xFF;

            (mask_lo | (mask_hi << 8)) as u64
        }
    }

    // ---- x86_64 AVX2: 16 × u16 (single 256-bit pass) ----

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn search_tags_u16_16_avx2(tags: &[u16], needle: u16) -> u64 {
        use std::arch::x86_64::*;
        // SAFETY: caller guarantees tags is 32-byte aligned and has >= 16 elements.
        unsafe {
            let data = _mm256_load_si256(tags.as_ptr().cast::<__m256i>());
            let splat = _mm256_set1_epi16(needle as i16);
            let cmp = _mm256_cmpeq_epi16(data, splat);
            let packed = _mm256_packs_epi16(cmp, _mm256_setzero_si256());
            let permuted = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);
            let mask = _mm256_movemask_epi8(permuted) as u32;
            (mask & 0xFFFF) as u64
        }
    }

    // ---- Scalar fallback ----

    #[inline]
    fn search_tags_scalar(tags: &[u8], needle: u8, ways: u64) -> u64 {
        let mut bits: u64 = 0;
        for (i, &tag) in tags.iter().enumerate().take(ways as usize) {
            if tag == needle {
                bits |= 1 << i;
            }
        }
        bits
    }

    #[inline]
    fn search_tags_u16_scalar(tags: &[u16], needle: u16, ways: u64) -> u64 {
        let mut bits: u64 = 0;
        for (i, &tag) in tags.iter().enumerate().take(ways as usize) {
            if tag == needle {
                bits |= 1 << i;
            }
        }
        bits
    }
}

enum TagStore {
    U8(AlignedBuf<u8>),
    U16(AlignedBuf<u16>),
}

impl TagStore {
    fn clear(&mut self) {
        match self {
            TagStore::U8(buf) => buf.fill(0),
            TagStore::U16(buf) => buf.fill(0),
        }
    }

    #[cfg(test)]
    fn all_zero(&self) -> bool {
        match self {
            TagStore::U8(buf) => buf.as_slice().iter().all(|&t| t == 0),
            TagStore::U16(buf) => buf.as_slice().iter().all(|&t| t == 0),
        }
    }

    /// # Safety
    /// Must only be called once, with the same allocator used for allocation.
    unsafe fn dealloc(&self, alloc: &impl Allocator) {
        match self {
            // SAFETY: forwarded from caller's safety contract.
            TagStore::U8(buf) => unsafe { buf.dealloc(alloc) },
            // SAFETY: forwarded from caller's safety contract.
            TagStore::U16(buf) => unsafe { buf.dealloc(alloc) },
        }
    }
}

/// Internal representation of a tag: either 8 or 16 bits, stored as u16.
type Tag = u16;

struct SetView {
    tag: Tag,
    offset: u64,
}

/// A set-associative cache with CLOCK eviction, SIMD tag matching, and custom allocator support.
pub struct SetAssociativeCache<E: KeyExtract, S: BuildHasher, L: CacheLayout, A: Allocator = Global>
{
    sets: u64,
    tag_store: TagStore,
    values: AlignedBuf<MaybeUninit<E::Value>>,
    counts: PackedArray,
    clocks: PackedArray,
    /// Cache hit/miss metrics.
    pub metrics: Metrics,
    hash_builder: S,
    alloc: A,
    _extract: PhantomData<E>,
    _layout: PhantomData<L>,
}

impl<E: KeyExtract, S: BuildHasher + Default, L: CacheLayout> SetAssociativeCache<E, S, L>
where
    E::Key: Hash + Eq,
{
    /// Create a new cache with default hasher and global allocator.
    pub fn new(value_count_max: u64) -> Self {
        Self::with_hasher(value_count_max, S::default())
    }
}

impl<E: KeyExtract, S: BuildHasher, L: CacheLayout> SetAssociativeCache<E, S, L>
where
    E::Key: Hash + Eq,
{
    /// Create a new cache with the given hasher and global allocator.
    pub fn with_hasher(value_count_max: u64, hash_builder: S) -> Self {
        Self::with_hasher_and_alloc(value_count_max, hash_builder, Global)
    }
}

impl<E: KeyExtract, S: BuildHasher, L: CacheLayout, A: Allocator> SetAssociativeCache<E, S, L, A>
where
    E::Key: Hash + Eq,
{
    /// Create a new cache with the given hasher and allocator.
    pub fn with_hasher_and_alloc(value_count_max: u64, hash_builder: S, alloc: A) -> Self {
        const { assert!(L::WAYS == 2 || L::WAYS == 4 || L::WAYS == 16) };
        const { assert!(L::TAG_BITS == 8 || L::TAG_BITS == 16) };
        const { assert!(L::CLOCK_BITS == 1 || L::CLOCK_BITS == 2 || L::CLOCK_BITS == 4) };
        const { assert!(L::CACHE_LINE_SIZE.is_power_of_two()) };

        let ways = L::WAYS;
        let sets = value_count_max / ways;
        let cache_line_size = L::CACHE_LINE_SIZE as usize;

        assert!(value_count_max > 0);
        assert!(value_count_max >= ways);
        assert!(value_count_max.is_multiple_of(ways));

        let value_count_max_multiple = Self::value_count_max_multiple();
        assert!(
            value_count_max.is_multiple_of(value_count_max_multiple),
            "value_count_max ({}) must be a multiple of {}",
            value_count_max,
            value_count_max_multiple,
        );

        // Tags: align for SIMD loads (SSE2=16, AVX2=32, AVX-512=64).
        let tag_align = cache_line_size.max(64);
        let tag_store = match L::TAG_BITS {
            8 => TagStore::U8(AlignedBuf::alloc_zeroed(
                value_count_max as usize,
                tag_align,
                &alloc,
            )),
            16 => TagStore::U16(AlignedBuf::alloc_zeroed(
                value_count_max as usize,
                tag_align,
                &alloc,
            )),
            _ => unreachable!(),
        };

        let values = AlignedBuf::<MaybeUninit<E::Value>>::alloc_zeroed(
            value_count_max as usize,
            cache_line_size,
            &alloc,
        );
        let counts = PackedArray::new_aligned(
            L::CLOCK_BITS as u32,
            value_count_max,
            cache_line_size,
            &alloc,
        );
        let clock_hand_bits = log2(L::WAYS);
        let clocks =
            PackedArray::new_aligned(clock_hand_bits as u32, sets, cache_line_size, &alloc);

        Self {
            sets,
            tag_store,
            values,
            counts,
            clocks,
            metrics: Metrics::default(),
            hash_builder,
            alloc,
            _extract: PhantomData,
            _layout: PhantomData,
        }
    }

    /// Minimum alignment multiple that `value_count_max` must satisfy.
    pub fn value_count_max_multiple() -> u64 {
        let cache_line_size = L::CACHE_LINE_SIZE;
        let ways = L::WAYS;
        let clock_bits = L::CLOCK_BITS;
        let value_size = std::mem::size_of::<E::Value>() as u64;
        let values_part =
            (value_size.max(cache_line_size) / value_size.min(cache_line_size)) * ways;
        let counts_part = (cache_line_size * 8) / clock_bits;
        values_part.max(counts_part)
    }

    /// Reset the cache, clearing all entries and metrics.
    pub fn reset(&mut self) {
        let total_slots = self.sets * L::WAYS;
        for i in 0..total_slots {
            if self.counts.get(i) > 0 {
                // SAFETY: count > 0 means the slot was initialized via MaybeUninit::write.
                unsafe { self.values.as_mut_slice()[i as usize].assume_init_drop() };
            }
        }
        self.tag_store.clear();
        self.counts.clear();
        self.clocks.clear();
        self.metrics = Metrics::default();
    }

    /// Look up a key, returning its index if found.
    pub fn get_index<Q>(&mut self, key: &Q) -> Option<usize>
    where
        Q: Hash + Equivalent<E::Key> + ?Sized,
    {
        let set = self.associate(key);
        if let Some(way) = self.search(&set, key) {
            self.metrics.hits += 1;
            let idx = set.offset + way as u64;
            let count = self.counts.get(idx);
            let max = (1u64 << L::CLOCK_BITS) - 1;
            self.counts.set(idx, count.saturating_add(1).min(max));
            Some(idx as usize)
        } else {
            self.metrics.misses += 1;
            None
        }
    }

    /// Look up a key, returning a reference to the value if found.
    pub fn get<Q>(&mut self, key: &Q) -> Option<&E::Value>
    where
        Q: Hash + Equivalent<E::Key> + ?Sized,
    {
        let index = self.get_index(key)?;
        // SAFETY: get_index only returns an index where count > 0, meaning the slot
        // was initialized via MaybeUninit::write in upsert.
        Some(unsafe { self.values.as_slice()[index].assume_init_ref() })
    }

    /// Look up a key, returning a mutable reference to the value if found.
    pub fn get_mut<Q>(&mut self, key: &Q) -> Option<&mut E::Value>
    where
        Q: Hash + Equivalent<E::Key> + ?Sized,
    {
        let index = self.get_index(key)?;
        // SAFETY: get_index only returns an index where count > 0, meaning the slot
        // was initialized via MaybeUninit::write in upsert.
        Some(unsafe { self.values.as_mut_slice()[index].assume_init_mut() })
    }

    /// Remove a key from the cache if present.
    pub fn remove<Q>(&mut self, key: &Q) -> Option<E::Value>
    where
        Q: Hash + Equivalent<E::Key> + ?Sized,
    {
        let set = self.associate(key);
        let way = self.search(&set, key)?;
        let idx = set.offset + way as u64;
        // SAFETY: search only returns a way where count > 0, meaning initialized.
        // assume_init_read moves the value out; we set count to 0 so the slot
        // is treated as uninitialized from here on.
        let removed = unsafe { self.values.as_slice()[idx as usize].assume_init_read() };
        self.counts.set(idx, 0);
        self.metrics.value_count -= 1;
        Some(removed)
    }

    /// Hint that the key is less likely to be accessed in the future.
    pub fn demote<Q>(&mut self, key: &Q)
    where
        Q: Hash + Equivalent<E::Key> + ?Sized,
    {
        let set = self.associate(key);
        if let Some(way) = self.search(&set, key) {
            self.counts.set(set.offset + way as u64, 1);
        }
    }

    /// Upsert a value, evicting an older entry if needed.
    pub fn upsert(&mut self, value: E::Value) -> UpsertResult<E::Value> {
        // Extract key twice (inline, cheap) to avoid requiring Key: Copy.
        // Each temporary borrow of `value` is released before the next statement.
        let set = self.associate(E::extract(&value));
        let existing_way = self.search(&set, E::extract(&value));

        if let Some(way) = existing_way {
            let idx = (set.offset + way as u64) as usize;
            self.counts.set(idx as u64, 1);
            let slot = &mut self.values.as_mut_slice()[idx];
            // SAFETY: search found this slot with count > 0, so it is initialized.
            let evicted = unsafe { slot.assume_init_read() };
            slot.write(value);
            return UpsertResult {
                index: idx,
                updated: UpdateOrInsert::Update,
                evicted: Some(evicted),
            };
        }

        let ways = L::WAYS;
        let max_count = (1u64 << L::CLOCK_BITS) - 1;
        let clock_index = set.offset / ways;

        let mut way = self.clocks.get(clock_index);
        let way_mask = ways - 1;

        // Maximum iterations: every slot at max count, decrementing all down to 1,
        // then one more iteration to decrement to 0 and break.
        let clock_iterations_max = ways * (max_count - 1);

        let mut evicted: Option<E::Value> = None;
        let mut safety_count = 0u64;
        loop {
            if safety_count > clock_iterations_max {
                unreachable!("CLOCK algorithm exceeded maximum iterations");
            }
            let idx = set.offset + way;
            let mut count = self.counts.get(idx);
            if count == 0 {
                break; // Way is already free.
            }
            count -= 1;
            self.counts.set(idx, count);
            if count == 0 {
                // SAFETY: count was > 0 before decrement, so the slot is initialized.
                evicted = Some(unsafe { self.values.as_slice()[idx as usize].assume_init_read() });
                break;
            }
            safety_count += 1;
            way = (way + 1) & way_mask;
        }

        debug_assert!(self.counts.get(set.offset + way) == 0);

        let idx = (set.offset + way) as usize;
        match &mut self.tag_store {
            TagStore::U8(buf) => buf.as_mut_slice()[idx] = set.tag as u8,
            TagStore::U16(buf) => buf.as_mut_slice()[idx] = set.tag,
        }
        self.values.as_mut_slice()[idx].write(value);
        self.counts.set(set.offset + way, 1);
        self.clocks.set(clock_index, (way + 1) & way_mask);
        if evicted.is_none() {
            self.metrics.value_count += 1;
        }

        UpsertResult {
            index: idx,
            updated: UpdateOrInsert::Insert,
            evicted,
        }
    }

    #[inline]
    fn associate<Q: Hash + ?Sized>(&self, key: &Q) -> SetView {
        let entropy = self.hash_builder.hash_one(key);
        let tag = (entropy & ((1u64 << L::TAG_BITS) - 1)) as Tag;
        let index = fastrange(entropy, self.sets);
        let offset = index * L::WAYS;
        SetView { tag, offset }
    }

    #[inline]
    fn search<Q>(&self, set: &SetView, key: &Q) -> Option<u16>
    where
        Q: Equivalent<E::Key> + ?Sized,
    {
        let ways = L::WAYS;
        let offset = set.offset;

        let matching_ways: u64 = match &self.tag_store {
            TagStore::U8(buf) => {
                let tags = buf.as_slice();
                let slice = &tags[offset as usize..(offset + ways) as usize];
                simd::search_tags(slice, set.tag as u8, ways)
            }
            TagStore::U16(buf) => {
                let tags = buf.as_slice();
                let slice = &tags[offset as usize..(offset + ways) as usize];
                simd::search_tags_u16(slice, set.tag, ways)
            }
        };

        if matching_ways == 0 {
            return None;
        }

        for way in 0..ways {
            if (matching_ways >> way) & 1 == 1 && self.counts.get(offset + way) > 0 {
                // SAFETY: count > 0 means the slot was initialized via MaybeUninit::write.
                let val =
                    unsafe { self.values.as_slice()[(offset + way) as usize].assume_init_ref() };
                if key.equivalent(E::extract(val)) {
                    return Some(way as u16);
                }
            }
        }
        None
    }
}

impl<E: KeyExtract, S: BuildHasher, L: CacheLayout, A: Allocator> Drop
    for SetAssociativeCache<E, S, L, A>
{
    fn drop(&mut self) {
        // Drop all live values before deallocating.
        let total_slots = self.sets * L::WAYS;
        for i in 0..total_slots {
            if self.counts.get(i) > 0 {
                // SAFETY: count > 0 means the slot was initialized via MaybeUninit::write.
                unsafe { self.values.as_mut_slice()[i as usize].assume_init_drop() };
            }
        }
        // SAFETY: each buffer is deallocated exactly once with the same allocator
        // that was used for allocation, stored in self.alloc.
        unsafe {
            self.tag_store.dealloc(&self.alloc);
            self.values.dealloc(&self.alloc);
            self.counts.dealloc(&self.alloc);
            self.clocks.dealloc(&self.alloc);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::Hasher;

    // --- PackedArray tests ---

    #[test]
    fn packed_array_unit() {
        let mut words = [0u64; 8];
        words[1] = 0b10110010;

        let mut p = PackedArray {
            uint_bits: 2,
            words: PackedWords::Vec(words.to_vec()),
        };

        assert_eq!(p.get(32 + 0), 0b10);
        assert_eq!(p.get(32 + 1), 0b00);
        assert_eq!(p.get(32 + 2), 0b11);
        assert_eq!(p.get(32 + 3), 0b10);

        p.set(0, 0b01);
        assert_eq!(p.words().to_vec()[0], 0b00000001);
        assert_eq!(p.get(0), 0b01);

        p.set(1, 0b10);
        assert_eq!(p.words().to_vec()[0], 0b00001001);
        assert_eq!(p.get(1), 0b10);

        p.set(2, 0b11);
        assert_eq!(p.words().to_vec()[0], 0b00111001);
        assert_eq!(p.get(2), 0b11);

        p.set(3, 0b11);
        assert_eq!(p.words().to_vec()[0], 0b11111001);
        assert_eq!(p.get(3), 0b11);

        p.set(3, 0b01);
        assert_eq!(p.words().to_vec()[0], 0b01111001);
        assert_eq!(p.get(3), 0b01);

        p.set(3, 0b00);
        assert_eq!(p.words().to_vec()[0], 0b00111001);
        assert_eq!(p.get(3), 0b00);

        p.set(4, 0b11);
        assert_eq!(
            p.words().to_vec()[0],
            0b0000000000000000000000000000000000000000000000000000001100111001
        );

        p.set(31, 0b11);
        assert_eq!(
            p.words().to_vec()[0],
            0b1100000000000000000000000000000000000000000000000000001100111001
        );
    }

    // --- BuildHasher implementations for tests ---

    /// A hasher that returns the u64 as-is (identity hash).
    struct IdentityHasher(u64);

    impl Hasher for IdentityHasher {
        fn finish(&self) -> u64 {
            self.0
        }
        fn write(&mut self, _bytes: &[u8]) {
            unimplemented!("IdentityHasher only supports write_u64");
        }
        fn write_u64(&mut self, i: u64) {
            self.0 = i;
        }
    }

    #[derive(Clone)]
    struct IdentityBuildHasher;

    impl BuildHasher for IdentityBuildHasher {
        type Hasher = IdentityHasher;
        fn build_hasher(&self) -> IdentityHasher {
            IdentityHasher(0)
        }
    }

    /// A hasher that always returns 0.
    struct ZeroHasher;

    impl Hasher for ZeroHasher {
        fn finish(&self) -> u64 {
            0
        }
        fn write(&mut self, _bytes: &[u8]) {}
        fn write_u64(&mut self, _i: u64) {}
    }

    #[derive(Clone)]
    struct ZeroBuildHasher;

    impl BuildHasher for ZeroBuildHasher {
        type Hasher = ZeroHasher;
        fn build_hasher(&self) -> ZeroHasher {
            ZeroHasher
        }
    }

    // --- SetAssociativeCache: KeyExtract for u64 identity ---

    struct IdentityExtract;

    impl KeyExtract for IdentityExtract {
        type Key = u64;
        type Value = u64;

        #[inline]
        fn extract(value: &u64) -> &u64 {
            value
        }
    }

    // --- Test layout types ---

    struct Ways2Layout;
    impl CacheLayout for Ways2Layout {
        const WAYS: u64 = 2;
        const TAG_BITS: u64 = 8;
        const CLOCK_BITS: u64 = 2;
        const CACHE_LINE_SIZE: u64 = 64;
    }

    struct Ways4Layout;
    impl CacheLayout for Ways4Layout {
        const WAYS: u64 = 4;
        const TAG_BITS: u64 = 8;
        const CLOCK_BITS: u64 = 2;
        const CACHE_LINE_SIZE: u64 = 64;
    }

    struct Tag16Layout;
    impl CacheLayout for Tag16Layout {
        const WAYS: u64 = 16;
        const TAG_BITS: u64 = 16;
        const CLOCK_BITS: u64 = 2;
        const CACHE_LINE_SIZE: u64 = 64;
    }

    struct Clock1Layout;
    impl CacheLayout for Clock1Layout {
        const WAYS: u64 = 16;
        const TAG_BITS: u64 = 8;
        const CLOCK_BITS: u64 = 1;
        const CACHE_LINE_SIZE: u64 = 64;
    }

    struct Clock4Layout;
    impl CacheLayout for Clock4Layout {
        const WAYS: u64 = 16;
        const TAG_BITS: u64 = 8;
        const CLOCK_BITS: u64 = 4;
        const CACHE_LINE_SIZE: u64 = 64;
    }

    fn run_cache_test_with_hasher<S: BuildHasher, L: CacheLayout>(hash_builder: S) {
        let ways = L::WAYS;
        let value_count_max = 16 * 16 * 8;

        let mut sac = SetAssociativeCache::<IdentityExtract, S, L>::with_hasher(
            value_count_max,
            hash_builder,
        );

        // Verify initial state
        assert!(sac.tag_store.all_zero());
        assert!(sac.counts.words().iter().all(|&w| w == 0));
        assert!(sac.clocks.words().iter().all(|&w| w == 0));
        assert_eq!(sac.metrics.value_count, 0);

        let clock_bits = L::CLOCK_BITS;
        let max_count = (1u64 << clock_bits) - 1;

        let count_after_get = max_count.min(2);

        // Fill up the first set entirely.
        for i in 0..ways {
            assert_eq!(sac.clocks.get(0), i);
            let key = i * sac.sets;
            sac.upsert(key);
            assert_eq!(sac.counts.get(i), 1);
            assert_eq!(*sac.get(&key).unwrap(), key);
            assert_eq!(sac.counts.get(i), count_after_get);
        }
        assert_eq!(sac.clocks.get(0), 0);
        assert_eq!(sac.metrics.value_count, ways);

        // Insert another element into the first set, causing key 0 to be evicted.
        {
            let key = ways * sac.sets;
            sac.upsert(key);
            assert_eq!(sac.counts.get(0), 1);
            assert_eq!(*sac.get(&key).unwrap(), key);
            assert_eq!(sac.counts.get(0), count_after_get);

            assert!(sac.get(&0).is_none());

            for i in 1..ways {
                assert_eq!(sac.counts.get(i), 1);
            }
            assert_eq!(sac.metrics.value_count, ways);
        }

        // Ensure removal works.
        {
            let remove_way = ways - 1;
            let key = remove_way * sac.sets;
            assert_eq!(*sac.get(&key).unwrap(), key);

            sac.remove(&key);
            assert!(sac.get(&key).is_none());
            assert_eq!(sac.counts.get(remove_way), 0);
            assert_eq!(sac.metrics.value_count, ways - 1);
        }

        sac.reset();

        assert!(sac.tag_store.all_zero());
        assert!(sac.counts.words().iter().all(|&w| w == 0));
        assert!(sac.clocks.words().iter().all(|&w| w == 0));
        assert_eq!(sac.metrics.value_count, 0);

        // Fill up the first set entirely, maxing out the count for each slot.
        for i in 0..ways {
            assert_eq!(sac.clocks.get(0), i);
            let key = i * sac.sets;
            sac.upsert(key);
            assert_eq!(sac.counts.get(i), 1);
            for j in 2..=max_count {
                assert_eq!(*sac.get(&key).unwrap(), key);
                assert_eq!(sac.counts.get(i), j);
            }
            // One more get should stay at max.
            assert_eq!(*sac.get(&key).unwrap(), key);
            assert_eq!(sac.counts.get(i), max_count);
        }
        assert_eq!(sac.clocks.get(0), 0);
        assert_eq!(sac.metrics.value_count, ways);

        // Insert another element into the first set, causing key 0 to be evicted.
        {
            let key = ways * sac.sets;
            sac.upsert(key);
            assert_eq!(sac.counts.get(0), 1);
            assert_eq!(*sac.get(&key).unwrap(), key);
            assert_eq!(sac.counts.get(0), count_after_get);

            assert!(sac.get(&0).is_none());

            for i in 1..ways {
                assert_eq!(sac.counts.get(i), 1);
            }
            assert_eq!(sac.metrics.value_count, ways);
        }
    }

    #[test]
    fn set_associative_cache_eviction() {
        run_cache_test_with_hasher::<_, DefaultLayout>(IdentityBuildHasher);
    }

    #[test]
    fn set_associative_cache_hash_collision() {
        run_cache_test_with_hasher::<_, DefaultLayout>(ZeroBuildHasher);
    }

    #[test]
    fn set_associative_cache_ways_2() {
        run_cache_test_with_hasher::<_, Ways2Layout>(IdentityBuildHasher);
    }

    #[test]
    fn set_associative_cache_ways_4() {
        run_cache_test_with_hasher::<_, Ways4Layout>(IdentityBuildHasher);
    }

    #[test]
    fn set_associative_cache_tag_bits_16() {
        run_cache_test_with_hasher::<_, Tag16Layout>(IdentityBuildHasher);
    }

    #[test]
    fn set_associative_cache_clock_bits_1() {
        run_cache_test_with_hasher::<_, Clock1Layout>(IdentityBuildHasher);
    }

    #[test]
    fn set_associative_cache_clock_bits_4() {
        run_cache_test_with_hasher::<_, Clock4Layout>(IdentityBuildHasher);
    }

    // --- SIMD search_tags correctness ---

    #[test]
    fn search_tags_correctness() {
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};

        let mut rng = SmallRng::seed_from_u64(42);

        // Aligned buffer on the stack for SIMD loads (32-byte aligned covers both SSE2 and AVX2).
        #[repr(C, align(32))]
        struct AlignedTags([u8; 16]);

        for ways in [2u64, 4, 16] {
            for _ in 0..10_000 {
                let mut buf = AlignedTags([0u8; 16]);
                let tags = &mut buf.0[..ways as usize];
                for t in tags.iter_mut() {
                    *t = rng.random();
                }
                let needle: u8 = rng.random();

                // Force some matches.
                let matches_min = rng.random_range(0..=ways as usize);
                let mut indices: Vec<usize> = (0..ways as usize).collect();
                // Simple Fisher-Yates
                for i in (1..indices.len()).rev() {
                    let j = rng.random_range(0..=i);
                    indices.swap(i, j);
                }
                for &idx in &indices[..matches_min] {
                    tags[idx] = needle;
                }

                // Reference
                let mut expected = 0u64;
                for (i, &t) in tags.iter().enumerate() {
                    if t == needle {
                        expected |= 1 << i;
                    }
                }

                let actual = simd::search_tags(&buf.0[..ways as usize], needle, ways);
                assert_eq!(
                    expected, actual,
                    "ways={ways} needle={needle} tags={:?}",
                    &buf.0[..ways as usize]
                );
            }
        }
    }

    // --- PairExtract test ---

    #[test]
    fn pair_extract_works() {
        type E = PairExtract<u32, String>;
        let val = (42u32, "hello".to_string());
        assert_eq!(E::extract(&val), &42u32);
    }
}
