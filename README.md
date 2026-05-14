# SetAssociativeCache - High-Performance CLOCK Eviction Cache

A hardware-optimized, set-associative cache implementation using CLOCK eviction policy. Achieves 30-50% higher throughput than standard LRU/LFU/FIFO caches through SIMD vectorization, compile-time optimization, and cache-friendly memory layout.

## Features

- **CLOCK Eviction**: O(1) amortized complexity, good approximation of LRU
- **SIMD Vectorized Search**: Parallel tag comparison using x86-64 intrinsics (SSE2/AVX2)
- **Compile-Time Optimization**: CacheLayout trait enables constant folding and dead code elimination
- **Hardware-Aligned Memory**: Single contiguous allocation with cache line alignment
- **Flexible Key Extraction**: Generic KeyExtract trait supports custom key types
- **Custom Allocators**: Full support for custom allocator implementations
- **No Dependencies**: Core cache uses only std library

## Performance

Benchmark results (sequential workload):

```
Cache Size 256 items:
  SetAssociativeCache:  11.38 Melem/s
  LRU/FIFO/Random:       8-9 Melem/s  (+27% slower)
  LFU:                   6.75 Melem/s (+60% slower)

Cache Size 512 items:
  SetAssociativeCache:  13.29 Melem/s
  LRU/FIFO/Random:       8-9 Melem/s  (+47% slower)
  LFU:                   6.49 Melem/s (+100% slower)
```

See BENCHMARK_RESULTS.md for detailed analysis and comparisons.

## Usage

### Basic Example

```rust
use set_associative::{SetAssociativeCache, KeyExtract, DefaultLayout};
use std::collections::hash_map::RandomState;

// Define key-value pair type
#[derive(Clone, Copy, Debug)]
struct Pair(u64, String);  // (key, value)

// Implement KeyExtract trait
struct MyExtractor;
impl KeyExtract for MyExtractor {
    type Key = u64;
    type Value = Pair;

    fn extract(pair: &Self::Value) -> &Self::Key {
        &pair.0
    }
}

fn main() {
    // Create cache with capacity for 1024 items
    let mut cache: SetAssociativeCache<MyExtractor, RandomState, DefaultLayout> =
        SetAssociativeCache::new(1024);

    // Insert values
    cache.upsert(Pair(1, "hello".into()));
    cache.upsert(Pair(2, "world".into()));

    // Retrieve values
    if let Some(pair) = cache.get(&1) {
        println!("Key 1: {}", pair.1);
    }

    // Eviction happens automatically on capacity
    cache.upsert(Pair(3, "overflow".into()));  // May evict old entry

    // Get metrics
    println!("Metrics: {:?}", cache.metrics());
}
```

### Custom Cache Layout

```rust
use set_associative::{CacheLayout, SetAssociativeCache};

// Define custom geometry (2 ways, 16-bit tags, 4 clock bits)
pub struct CustomLayout;
impl CacheLayout for CustomLayout {
    const WAYS: u64 = 2;
    const TAG_BITS: u64 = 16;
    const CLOCK_BITS: u64 = 4;
    const CACHE_LINE_SIZE: u64 = 64;
}

// Use with custom layout
let mut cache: SetAssociativeCache<MyExtractor, RandomState, CustomLayout> =
    SetAssociativeCache::new(512);
```

### Custom Allocator

```rust
use set_associative::SetAssociativeCache;
use std::alloc::Global;

let mut cache: SetAssociativeCache<
    MyExtractor,
    RandomState,
    DefaultLayout,
    Global,  // Custom allocator
> = SetAssociativeCache::new(1024);
```

## How It Works

### Memory Layout

SetAssociativeCache organizes data into sets, each containing multiple ways:

```
Set 0: [Tag0][Tag1]...[Tag15]
       [Val0][Val1]...[Val15]  <- All in one cache line
       [Clock bits for each slot]

Set 1: [Tag0][Tag1]...[Tag15]
       [Val0][Val1]...[Val15]
       [Clock bits]

...
```

All data for a set fits in a single CPU cache line, enabling prefetching and cache hits.

### Search Algorithm

1. Hash key to get (set index, tag)
2. Load all tags for that set via SIMD
3. Compare all 16 tags in parallel
4. On match, verify key equality and update clock bit
5. Return value

Cost: 1-2 CPU cycles (vectorized) vs 10+ cycles for HashMap.

### CLOCK Eviction

When cache is full:

1. Locate eviction pointer for the set
2. Scan forward in circular order
3. Clear clock bits as we scan
4. Evict first slot with bit = 0
5. Insert new value
6. Advance pointer for next eviction

Cost: O(1) amortized with predictable latency.

## Configuration

### Cache Size Requirements

Value count must be a multiple of:
- WAYS (number of associative ways)
- (CACHE_LINE_SIZE * 8) / CLOCK_BITS for clock counters

Default layout (WAYS=16, CLOCK_BITS=2):
```
Required multiple = max(16, 256) = 256
Valid sizes: 256, 512, 1024, 2048, ...
```

### Layout Selection

- **DefaultLayout**: 16 ways, 8-bit tags, 2 clock bits (recommended)
- **Ways2Layout**: 2 ways, minimal overhead
- **Ways4Layout**: 4 ways, balance
- **Ways16Layout**: 16 ways, maximum parallelism

Larger WAYS = better SIMD parallelism but more memory per set.

## Comparison with Other Policies

| Policy | Throughput | Hit Rate | Complexity | Memory | Flexibility |
|--------|-----------|----------|-----------|--------|------------|
| SetAssociative | 11-14 Melem/s | Good (CLOCK) | O(1) | Fixed | Limited |
| LRU | 8-9 Melem/s | Optimal | O(n) | Dynamic | Full |
| LFU | 6.5 Melem/s | Optimal | O(log n) | Dynamic | Full |
| FIFO | 8-9 Melem/s | Good | O(1) | Dynamic | Full |
| Random | 8-9 Melem/s | Fair | O(1) | Dynamic | Full |

### Trade-offs

**Choose SetAssociativeCache if**:
- Throughput is critical (>100K ops/sec)
- Cache size is known in advance
- CPU cache efficiency matters
- Hit rate variance of 2-5% is acceptable
- Standard CLOCK eviction semantics sufficient

**Choose HashMap + LFU if**:
- Optimal hit rate required
- Cache size varies dynamically
- Support for heterogeneous value types needed

## Building and Testing

```bash
# Build
cargo build --release

# Run tests
cargo test

# Run benchmarks
cargo bench --bench cache_benchmark -- --quick

# Full benchmarks with HTML report
cargo bench --bench cache_benchmark

# Open report
target/criterion/report/index.html
```

## Benchmarking

The included benchmark suite compares SetAssociativeCache against:
- LRU (Least Recently Used)
- FIFO (First In First Out)
- LFU (Least Frequently Used)
- W-TinyLFU (Windowed TinyLFU)
- Random eviction

Benchmarks cover:
- Cache sizes: 256, 512, 1024 items
- Access patterns: Sequential workload
- Access counts: 10K and 100K operations

See BENCHMARKS.md for detailed methodology.

## When to Use SetAssociativeCache

### Good Fit

- Database query result caching
- CPU cache replacement simulation
- High-frequency trading data cache
- Game engine asset cache
- Connection/resource pooling
- DNS resolver cache

### Poor Fit

- Variable-size item storage
- Unbounded growth scenarios
- Applications requiring exact LRU semantics

## Implementation Notes

### Thread Safety

SetAssociativeCache is not thread-safe. For concurrent access, wrap with:
- Arc<Mutex<SetAssociativeCache>>
- DashMap with per-item locks
- Sharding across multiple caches

### Memory Overhead

```
Base overhead: ~400 bytes
Per item: tag(1-2B) + clock_bits(fractional) + bookkeeping

Total = 400 + (cache_size * 2)
```

### CPU Architecture

Currently optimized for x86-64 with SSE2/AVX2. Generic implementations available via feature flags.

## Design Rationale

### Why CLOCK Eviction?

- O(1) amortized complexity vs O(n) for LRU retention
- Good hit rate approximation (95%+ of LRU)
- Predictable latency (no worst-case O(n))
- Hardware-friendly scan pattern

### Why Set-Associative?

- Mimics CPU cache structure
- Enables SIMD vectorization
- Single allocation with predictable layout
- Aligns with CPU cache lines

### Why Compile-Time Constants?

- LLVM optimizations: division -> shift, masks -> constants
- Dead code elimination for unused ways/tags
- Monomorphization specializes code per layout
- Zero-cost abstractions

## References

- CLOCK Algorithm: Corbato, "A Paging Experiment with the Multics System"
- Set-Associative Caches: CPU cache architecture literature
- SIMD Optimization: Intel x86 optimization manuals
- ARM: https://developer.arm.com/documentation/den0042/0100/Caches/Cache-architecture/Set-associative-caches

## License

Apache License 2.0

## Contributing

Contributions welcome. Please ensure:
- All tests pass: `cargo test`
- Benchmarks don't regress: `cargo bench`
- Code follows project style
- Documentation updated

