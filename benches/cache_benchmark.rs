use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::collections::HashMap;
use std::collections::VecDeque;
use lfu_cache::LfuCache as LfuCacheImpl;
use wtinylfu::WTinyLfuCache;

use set_associative::{KeyExtract, SetAssociativeCache, DefaultLayout};

/// Simple key-value pair for benchmarking
#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
struct Key(u64);

#[derive(Clone, Copy, Debug)]
struct Value(u64);

/// Pair for SetAssociativeCache
#[derive(Clone, Copy, Debug)]
struct Pair(Key, Value);

/// KeyExtract implementation for Pair
struct PairExtractor;

impl KeyExtract for PairExtractor {
    type Key = Key;
    type Value = Pair;

    fn extract(pair: &Self::Value) -> &Self::Key {
        &pair.0
    }
}

/// LRU cache using HashMap + VecDeque for tracking access order
struct LruCache {
    cache: HashMap<Key, Value>,
    access_order: VecDeque<Key>,
    max_size: usize,
}

impl LruCache {
    fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(max_size),
            access_order: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    fn get(&mut self, key: &Key) -> Option<Value> {
        if let Some(&value) = self.cache.get(key) {
            // Move to end (most recently used)
            self.access_order.retain(|k| k != key);
            self.access_order.push_back(*key);
            Some(value)
        } else {
            None
        }
    }

    fn insert(&mut self, key: Key, value: Value) {
        if self.cache.contains_key(&key) {
            self.cache.insert(key, value);
            self.access_order.retain(|k| k != &key);
            self.access_order.push_back(key);
        } else {
            if self.cache.len() >= self.max_size {
                // Evict least recently used
                if let Some(lru_key) = self.access_order.pop_front() {
                    self.cache.remove(&lru_key);
                }
            }
            self.cache.insert(key, value);
            self.access_order.push_back(key);
        }
    }
}

/// FIFO cache (simple queue-based eviction)
struct FifoCache {
    cache: HashMap<Key, Value>,
    insertion_order: VecDeque<Key>,
    max_size: usize,
}

impl FifoCache {
    fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(max_size),
            insertion_order: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    fn get(&self, key: &Key) -> Option<Value> {
        self.cache.get(key).copied()
    }

    fn insert(&mut self, key: Key, value: Value) {
        if !self.cache.contains_key(&key) {
            if self.cache.len() >= self.max_size {
                // Evict first inserted
                if let Some(fifo_key) = self.insertion_order.pop_front() {
                    self.cache.remove(&fifo_key);
                }
            }
            self.insertion_order.push_back(key);
        }
        self.cache.insert(key, value);
    }
}

/// Random eviction cache
struct RandomCache {
    cache: HashMap<Key, Value>,
    keys: Vec<Key>,
    max_size: usize,
}

impl RandomCache {
    fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(max_size),
            keys: Vec::with_capacity(max_size),
            max_size,
        }
    }

    fn get(&self, key: &Key) -> Option<Value> {
        self.cache.get(key).copied()
    }

    fn insert(&mut self, key: Key, value: Value) {
        if !self.cache.contains_key(&key) {
            if self.cache.len() >= self.max_size {
                // Evict random
                use rand::Rng as _;
                let mut rng = rand::rng();
                let idx = rng.random_range(0..self.keys.len());
                let victim = self.keys.swap_remove(idx);
                self.cache.remove(&victim);
            }
            self.keys.push(key);
        }
        self.cache.insert(key, value);
    }
}

/// Wrapper around LFU cache for uniform benchmark interface
struct LfuCache {
    cache: LfuCacheImpl<Key, Value>,
}

impl LfuCache {
    fn new(max_size: usize) -> Self {
        Self {
            cache: LfuCacheImpl::with_capacity(max_size),
        }
    }

    fn get(&mut self, key: &Key) -> Option<Value> {
        self.cache.get(key).copied()
    }

    fn insert(&mut self, key: Key, value: Value) {
        self.cache.insert(key, value);
    }
}

/// Wrapper around W-TinyLFU cache for uniform benchmark interface
struct WTinyLfuCacheWrapper {
    cache: WTinyLfuCache<Key, Value>,
}

impl WTinyLfuCacheWrapper {
    fn new(max_size: usize) -> Self {
        // W-TinyLFU window size is typically 10% of cache size
        let window_size = (max_size + 9) / 10;
        Self {
            cache: WTinyLfuCache::new(max_size, window_size),
        }
    }

    fn get(&mut self, key: &Key) -> Option<Value> {
        self.cache.get(key).copied()
    }

    fn insert(&mut self, key: Key, value: Value) {
        self.cache.push(key, value);
    }
}

/// Wrapper around SetAssociativeCache for uniform benchmark interface
struct SetAssocCache {
    cache: SetAssociativeCache<PairExtractor, std::collections::hash_map::RandomState, DefaultLayout>,
}

impl SetAssocCache {
    fn new(max_size: usize) -> Self {
        Self {
            cache: SetAssociativeCache::new(max_size as u64),
        }
    }

    fn get(&mut self, key: &Key) -> Option<Value> {
        self.cache.get(key).map(|pair| pair.1)
    }

    fn insert(&mut self, key: Key, value: Value) {
        let pair = Pair(key, value);
        self.cache.upsert(pair);
    }
}

fn workload_sequential(cache_size: usize, access_count: usize) {
    use rand::Rng as _;

    let mut rng = rand::rng();

    // Warm up keys — working set much larger than cache
    let working_set_size = cache_size * 4;

    let mut lru = LruCache::new(cache_size);
    let mut fifo = FifoCache::new(cache_size);
    let mut random = RandomCache::new(cache_size);
    let mut set_assoc = SetAssocCache::new(cache_size);

    for i in 0..access_count {
        let key = Key((i % working_set_size) as u64);
        let value = Value(rng.random());

        black_box(lru.get(&key));
        black_box(lru.insert(key, value));

        black_box(fifo.get(&key));
        black_box(fifo.insert(key, value));

        black_box(random.get(&key));
        black_box(random.insert(key, value));

        black_box(set_assoc.get(&key));
        black_box(set_assoc.insert(key, value));
    }
}

fn workload_random(cache_size: usize, access_count: usize) {
    use rand::Rng as _;

    let mut rng = rand::rng();
    let working_set_size = cache_size * 4;

    let mut lru = LruCache::new(cache_size);
    let mut fifo = FifoCache::new(cache_size);
    let mut random = RandomCache::new(cache_size);
    let mut set_assoc = SetAssocCache::new(cache_size);

    for _ in 0..access_count {
        let key = Key(rng.random_range(0..working_set_size as u64));
        let value = Value(rng.random());

        black_box(lru.get(&key));
        black_box(lru.insert(key, value));

        black_box(fifo.get(&key));
        black_box(fifo.insert(key, value));

        black_box(random.get(&key));
        black_box(random.insert(key, value));

        black_box(set_assoc.get(&key));
        black_box(set_assoc.insert(key, value));
    }
}

fn workload_skewed(cache_size: usize, access_count: usize) {
    use rand::Rng as _;

    let mut rng = rand::rng();
    let working_set_size = cache_size * 4;

    let mut lru = LruCache::new(cache_size);
    let mut fifo = FifoCache::new(cache_size);
    let mut random = RandomCache::new(cache_size);
    let mut set_assoc = SetAssocCache::new(cache_size);

    // Zipfian-like: access first 20% of keys 80% of the time
    for _ in 0..access_count {
        let key = if rng.random::<f64>() < 0.8 {
            Key(rng.random_range(0..(working_set_size / 5)) as u64)
        } else {
            Key(rng.random_range((working_set_size / 5)..working_set_size as usize) as u64)
        };
        let value = Value(rng.random());

        black_box(lru.get(&key));
        black_box(lru.insert(key, value));

        black_box(fifo.get(&key));
        black_box(fifo.insert(key, value));

        black_box(random.get(&key));
        black_box(random.insert(key, value));

        black_box(set_assoc.get(&key));
        black_box(set_assoc.insert(key, value));
    }
}

fn benchmark_lru(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_type/LRU");
    group.sample_size(10);
    let cache_sizes = vec![256, 512, 1024];
    let access_counts = vec![10_000, 100_000];

    for cache_size in &cache_sizes {
        for access_count in &access_counts {
            let bench_name = format!("sequential_{}_accesses_{}", cache_size, access_count);
            group.throughput(Throughput::Elements(*access_count as u64));
            group.bench_with_input(BenchmarkId::new("seq", bench_name), &(), |b, _| {
                b.iter(|| {
                    use rand::Rng as _;
                    let mut rng = rand::rng();
                    let working_set_size = cache_size * 4;
                    let mut lru = LruCache::new(*cache_size);
                    for i in 0..*access_count {
                        let key = Key((i % working_set_size) as u64);
                        let value = Value(rng.random());
                        black_box(lru.get(&key));
                        black_box(lru.insert(key, value));
                    }
                });
            });
        }
    }
    group.finish();
}

fn benchmark_fifo(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_type/FIFO");
    group.sample_size(10);
    let cache_sizes = vec![256, 512, 1024];
    let access_counts = vec![10_000, 100_000];

    for cache_size in &cache_sizes {
        for access_count in &access_counts {
            let bench_name = format!("sequential_{}_accesses_{}", cache_size, access_count);
            group.throughput(Throughput::Elements(*access_count as u64));
            group.bench_with_input(BenchmarkId::new("seq", bench_name), &(), |b, _| {
                b.iter(|| {
                    use rand::Rng as _;
                    let mut rng = rand::rng();
                    let working_set_size = cache_size * 4;
                    let mut fifo = FifoCache::new(*cache_size);
                    for i in 0..*access_count {
                        let key = Key((i % working_set_size) as u64);
                        let value = Value(rng.random());
                        black_box(fifo.get(&key));
                        black_box(fifo.insert(key, value));
                    }
                });
            });
        }
    }
    group.finish();
}

fn benchmark_lfu(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_type/LFU");
    group.sample_size(10);
    let cache_sizes = vec![256, 512, 1024];
    let access_counts = vec![10_000, 100_000];

    for cache_size in &cache_sizes {
        for access_count in &access_counts {
            let bench_name = format!("sequential_{}_accesses_{}", cache_size, access_count);
            group.throughput(Throughput::Elements(*access_count as u64));
            group.bench_with_input(BenchmarkId::new("seq", bench_name), &(), |b, _| {
                b.iter(|| {
                    use rand::Rng as _;
                    let mut rng = rand::rng();
                    let working_set_size = cache_size * 4;
                    let mut lfu = LfuCache::new(*cache_size);
                    for i in 0..*access_count {
                        let key = Key((i % working_set_size) as u64);
                        let value = Value(rng.random());
                        black_box(lfu.get(&key));
                        black_box(lfu.insert(key, value));
                    }
                });
            });
        }
    }
    group.finish();
}

fn benchmark_w_tinylfu(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_type/W-TinyLFU");
    group.sample_size(10);
    let cache_sizes = vec![256, 512, 1024];
    let access_counts = vec![10_000, 100_000];

    for cache_size in &cache_sizes {
        for access_count in &access_counts {
            let bench_name = format!("sequential_{}_accesses_{}", cache_size, access_count);
            group.throughput(Throughput::Elements(*access_count as u64));
            group.bench_with_input(BenchmarkId::new("seq", bench_name), &(), |b, _| {
                b.iter(|| {
                    use rand::Rng as _;
                    let mut rng = rand::rng();
                    let working_set_size = cache_size * 4;
                    let mut w_tinylfu = WTinyLfuCacheWrapper::new(*cache_size);
                    for i in 0..*access_count {
                        let key = Key((i % working_set_size) as u64);
                        let value = Value(rng.random());
                        black_box(w_tinylfu.get(&key));
                        black_box(w_tinylfu.insert(key, value));
                    }
                });
            });
        }
    }
    group.finish();
}

fn benchmark_random_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_type/Random");
    group.sample_size(10);
    let cache_sizes = vec![256, 512, 1024];
    let access_counts = vec![10_000, 100_000];

    for cache_size in &cache_sizes {
        for access_count in &access_counts {
            let bench_name = format!("sequential_{}_accesses_{}", cache_size, access_count);
            group.throughput(Throughput::Elements(*access_count as u64));
            group.bench_with_input(BenchmarkId::new("seq", bench_name), &(), |b, _| {
                b.iter(|| {
                    use rand::Rng as _;
                    let mut rng = rand::rng();
                    let working_set_size = cache_size * 4;
                    let mut random = RandomCache::new(*cache_size);
                    for i in 0..*access_count {
                        let key = Key((i % working_set_size) as u64);
                        let value = Value(rng.random());
                        black_box(random.get(&key));
                        black_box(random.insert(key, value));
                    }
                });
            });
        }
    }
    group.finish();
}

fn benchmark_set_associative(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_type/SetAssociative");
    group.sample_size(10);
    let cache_sizes = vec![256, 512, 1024];
    let access_counts = vec![10_000, 100_000];

    for cache_size in &cache_sizes {
        for access_count in &access_counts {
            let bench_name = format!("sequential_{}_accesses_{}", cache_size, access_count);
            group.throughput(Throughput::Elements(*access_count as u64));
            group.bench_with_input(BenchmarkId::new("seq", bench_name), &(), |b, _| {
                b.iter(|| {
                    use rand::Rng as _;
                    let mut rng = rand::rng();
                    let working_set_size = cache_size * 4;
                    let mut set_assoc = SetAssocCache::new(*cache_size);
                    for i in 0..*access_count {
                        let key = Key((i % working_set_size) as u64);
                        let value = Value(rng.random());
                        black_box(set_assoc.get(&key));
                        black_box(set_assoc.insert(key, value));
                    }
                });
            });
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_lru,
    benchmark_fifo,
    benchmark_lfu,
    benchmark_w_tinylfu,
    benchmark_random_cache,
    benchmark_set_associative
);
criterion_main!(benches);
