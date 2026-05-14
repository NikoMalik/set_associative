# SetAssociativeCache Algorithm - Detailed Explanation

Complete walkthrough of how SetAssociativeCache works: memory layout, search algorithm, eviction policy, and SIMD optimization.

## Table of Contents

1. [Memory Layout](#memory-layout)
2. [Hash Function](#hash-function)
3. [Search Algorithm](#search-algorithm)
4. [SIMD Tag Matching](#simd-tag-matching)
5. [CLOCK Eviction](#clock-eviction)
6. [Step-by-Step Example](#step-by-step-example)
7. [Performance Analysis](#performance-analysis)

## Memory Layout

### Physical Organization

SetAssociativeCache organizes all data into a single contiguous buffer divided into sets, each containing multiple ways:

```
┌─────────────────────────────────────────────────────┐
│  Set 0                                              │
├─────────────────────────────────────────────────────┤
│  Tags:   [Tag0][Tag1][Tag2]...[Tag15]  (16 bytes)  │
│  Values: [Val0][Val1][Val2]...[Val15]  (variable)  │
│  Counts: [CC0] [CC1] ...               (packed)    │
│  Clocks: [CK0] [CK1] ...               (packed)    │
├─────────────────────────────────────────────────────┤
│  Set 1                                              │
├─────────────────────────────────────────────────────┤
│  Tags:   [Tag0][Tag1]...[Tag15]                    │
│  Values: [Val0][Val1]...[Val15]                    │
│  Counts: [CC0] [CC1] ...                           │
│  Clocks: [CK0] [CK1] ...                           │
├─────────────────────────────────────────────────────┤
│  ...                                                │
└─────────────────────────────────────────────────────┘
```

### Structure Details

#### Tags Array

- Size: WAYS * 1 or 2 bytes (usually 16 tags = 16-32 bytes)
- Content: Hash bits extracted from key
- Purpose: Quick comparison without accessing values

```
Tag layout (64-bit key):
┌─────────────────────────────────┐
│   Hash value (from key)         │
└─────────────────────────────────┘
│ Set Index  │ Tag    │ Unused   │
│ (log2(N))  │ (8-16) │          │
└─────────────────────────────────┘

Example: 1M items cache
Sets = 1M / 16 = 65536
Set bits = 16 bits
Tag bits = 8 bits
```

#### Values Array

- Size: WAYS * sizeof(T) per set
- Content: Actual cached values
- Alignment: Proper alignment for SIMD loads

#### Counts Array

- Size: WAYS * CLOCK_BITS bits per set
- Content: Clock tick counters (when item was accessed)
- Purpose: Implement CLOCK eviction
- Packed: Multiple counters per byte

```
Example with CLOCK_BITS=2:
4 counters per byte

Byte: [CC0][CC1][CC2][CC3]
      │ 2  │ 2  │ 2  │ 2  │  bits
      └────┴────┴────┴────┘
```

#### Clocks Array

- Size: Circular pointer per set (log2(WAYS) bits)
- Content: Current eviction hand position
- Purpose: CLOCK algorithm pointer

### Memory Alignment

All data aligned to CPU cache line (typically 64 bytes):

```
┌────────────────────────────────────────────────────────────┐
│ Set 0 (fits in single cache line)                          │
├────────────────────────────────────────────────────────────┤
│ Tags (16B) | Values (28B) | Counts (1B) | Clocks (2B)     │
└────────────────────────────────────────────────────────────┘
               ^
         Prefetcher loads this entire line in one operation
```

Benefits:
- Single L1 cache hit retrieves all tags
- Prefetcher works optimally
- No pointer chasing
- SIMD can load all tags at once

## Hash Function

### Hash Computation

```rust
fn hash_key(key: &K, hash_builder: &S) -> u64 {
    hash_builder.hash_one(key)  // SipHash or similar
}
```

Input: Key of any type
Output: 64-bit hash value

### Hash Extraction

From 64-bit hash, extract:

```rust
let hash = hash_builder.hash_one(key);

// Extract set index (lower bits)
let set_bits = (64 - log2(num_sets)) as u32;
let set_index = hash & ((1u64 << set_bits) - 1);

// Extract tag (next bits)
let tag_bits = L::TAG_BITS as u32;
let tag_shift = set_bits;
let tag = (hash >> tag_shift) & ((1u64 << tag_bits) - 1);
```

Example with 65536 sets (16 bits), 8-bit tags:

```
Hash value:  0x_AABBCCDD_EEFF0011
             ││││││││ ││││││││ ││││││││ ││││││││
             V          V                 V

          Tag bits   Set bits
          (bits 16-23) (bits 0-15)

set_index = 0x0011 (bits 0-15)
tag = 0xFF (bits 16-23)
```

### Why This Works

- **Set index**: Lower bits, most random, distribute evenly
- **Tag**: Next bits, decorrelated from set index
- **Multiple hash functions**: Not needed, one hash sufficient

## Search Algorithm

### Overview

Goal: Find value associated with key in cache

Steps:
1. Hash the key
2. Compute set index and tag
3. Load all tags for that set
4. Compare all tags in parallel (SIMD)
5. Verify key equality for matching tag
6. Update clock bit on hit
7. Return value or None

### Pseudocode

```rust
fn get(&mut self, key: &K) -> Option<&V> {
    // Step 1: Hash the key
    let hash = self.hash_builder.hash_one(key);

    // Step 2: Extract set index and tag
    let set = Self::set_from_hash(hash);
    let tag = Self::tag_from_hash(hash);

    // Step 3: Load tags for this set
    let set_tags = self.tags.load_set(set);

    // Step 4: SIMD compare all tags
    let matches = simd_cmpeq(set_tags, tag);

    // Step 5: Check each matching tag
    for way in 0..L::WAYS {
        if matches[way] {
            let slot = set * L::WAYS + way;

            // Verify key equality
            let stored_key = E::extract(&self.values[slot]);
            if equivalent(key, stored_key) {
                // Step 6: Update clock bit
                self.clocks.set_bit(slot);

                // Step 7: Return value
                return Some(&self.values[slot]);
            }
        }
    }

    // Not found
    None
}
```

### Execution Timeline

```
Cycle 0:    hash_builder.hash_one(key)
            ├─ SipHash-2-4: ~20 cycles internally
            │  (pipelined with other instructions)
            └─ result: 64-bit hash

Cycle 1:    Extract set and tag
            ├─ set = hash & mask
            └─ tag = hash >> shift

Cycle 2:    Load tags
            ├─ Memory read to tags[set * WAYS]
            └─ (L1 cache hit expected)

Cycle 3:    SIMD compare
            ├─ _mm_cmpeq_epi8(loaded_tags, target_tag)
            └─ Result: bit vector of matches

Cycle 4:    Iterate matches
            ├─ for way in 0..16:
            │   if matches[way]:
            │     key_equality_check
            └─ Found or continue

Cycle 5:    On match:
            ├─ Update clock bit
            │  clocks.set_bit(slot)
            └─ Return &value[slot]
```

**Total: ~2-3 cycles on hit** (after hash computed)

## SIMD Tag Matching

### The Problem Without SIMD

Without SIMD, comparing 16 tags sequentially:

```rust
let mut found = false;
for way in 0..16 {
    if tags[way] == target_tag {
        found = true;
        break;
    }
}
// Cost: 1-16 comparisons, ~16 CPU cycles
```

Limitations:
- Sequential comparisons
- Branch prediction on each compare
- Cannot exploit CPU parallelism

### The Solution With SIMD

Load 16 tags and compare all at once:

```rust
// SSE2 version (128-bit registers, 16 x 8-bit values)
let tags_vector = _mm_loadu_si128(tags_ptr as *const __m128i);
let target_vector = _mm_set1_epi8(target_tag as i8);
let matches = _mm_cmpeq_epi8(tags_vector, target_vector);
// Result: 128-bit vector with 0xFF for matches, 0x00 for non-matches

// Extract result
let result = _mm_movemask_epi8(matches) as u16;
// result bit i = 1 if tags[i] == target_tag
```

### Instruction Details

#### SSE2 (128-bit)

```
Tags layout (memory):
[Tag0][Tag1][Tag2][Tag3][Tag4][Tag5]...[Tag15]
 1B    1B    1B    1B    1B    1B      1B

Load to XMM register:
_mm_loadu_si128() -> [Tag0 Tag1 Tag2 ... Tag15]

Comparison:
_mm_cmpeq_epi8(tags, target) produces:
[0xFF 0x00 0xFF 0x00 0x00 ...] (matches in 0xFF)

Extract scalar result:
_mm_movemask_epi8() -> 0b0101010100... (bit pattern)
```

#### AVX2 (256-bit, optional)

```
Can process 32 x 8-bit tags simultaneously:
_mm256_cmpeq_epi8(tags, target)
_mm256_movemask_epi8()

Even faster for larger ways or smaller tags.
```

### Cost Analysis

**Sequential**: O(16) = 16 comparisons, 16 cycles
**SIMD**: O(1) = 1 instruction, 1-2 cycles

**Speedup: 8-16x** on tag comparison alone.

## CLOCK Eviction

### The CLOCK Algorithm

When cache is full and need to insert new item:

1. Maintain circular pointer per set (0 to WAYS-1)
2. Scan forward in circular order
3. At each slot, check clock bit
4. If bit = 0, evict this slot
5. If bit = 1, clear it and continue
6. Advance pointer for next eviction

### Visual Example

16-way cache, hand pointing to slot 3:

```
Initial state (all clock bits = 1 after recent accesses):
┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│1 │1 │1 │>1│1 │0 │1 │1 │1 │0 │1 │1 │1 │1 │1 │1 │
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15

Scan from position 3:
Step 1: Check slot 3 -> bit=1, clear it and continue
┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│1 │1 │1 │>0│1 │0 │1 │1 │1 │0 │1 │1 │1 │1 │1 │1 │
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘

Step 2: Check slot 4 -> bit=1, clear it and continue
┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│1 │1 │1 │0 │>0│0 │1 │1 │1 │0 │1 │1 │1 │1 │1 │1 │
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘

Step 3: Check slot 5 -> bit=0, EVICT this slot!
┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│1 │1 │1 │0 │0 │[NEW]1 │1 │1 │0 │1 │1 │1 │1 │1 │1 │
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘

Set hand pointer to next position (6):
┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│1 │1 │1 │0 │0 │1 │>1│1 │1 │0 │1 │1 │1 │1 │1 │1 │
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
```

### Pseudocode

```rust
fn find_victim(&mut self, set: u64) -> u64 {
    let mut hand = self.clock_hand[set];

    loop {
        let slot = set * L::WAYS + hand;
        let bit = self.clocks.get_bit(slot);

        if bit == 0 {
            // Found victim
            self.clock_hand[set] = (hand + 1) % L::WAYS;
            return slot;
        }

        // Clear bit and advance
        self.clocks.clear_bit(slot);
        hand = (hand + 1) % L::WAYS;
    }
}

fn upsert(&mut self, value: V) -> Option<V> {
    let key = E::extract(&value);
    let set = self.set_from_key(key);

    // Search for existing entry
    if let Some(idx) = self.search_set(set, key) {
        // Replace existing
        let old = self.values[idx];
        self.values[idx] = value;
        return Some(old);
    }

    // Not found, need to evict
    let victim_slot = self.find_victim(set);
    let evicted = self.values[victim_slot];

    // Insert new value
    self.values[victim_slot] = value;
    self.clocks.set_bit(victim_slot);

    Some(evicted)
}
```

### Cost Analysis

**Best case** (find victim immediately):
- Check 1 slot, bit = 0
- Cost: 1-2 CPU cycles

**Average case** (need to clear 2-3 bits):
- Check 2-3 slots, clear bits, find one with bit = 0
- Cost: 3-5 CPU cycles

**Worst case** (all bits = 1):
- Scan through all WAYS slots, clearing each
- Cost: WAYS cycles (typically 16 = 16 cycles)
- But this is amortized (happens rarely)

**Amortized O(1)**: Average cost per eviction stays constant.

## Step-by-Step Example

### Initial State

Cache configuration:
```
WAYS = 4
TAG_BITS = 8
CLOCK_BITS = 2 (4 states per counter)
Items: (Key, Value)
```

Empty cache:
```
Set 0:  Tags: [0    0    0    0   ]
        Clks: [00   00   00   00  ]  (2 bits each)
        Hand: 0

Set 1:  Tags: [0    0    0    0   ]
        Clks: [00   00   00   00  ]
        Hand: 0
```

### Insert First Item

Command: `cache.upsert((Key(5), Value(100)))`

**Step 1: Hash**
```
hash = hash_builder.hash_one(5) = 0xABCD1234
set_index = 0xABCD1234 & 0x1 = 0  (assuming 2 sets)
tag = (0xABCD1234 >> 1) & 0xFF = 0x9A
```

**Step 2: Search Set 0**
```
Load tags: [0x00, 0x00, 0x00, 0x00]
SIMD compare with 0x9A: no matches
Miss!
```

**Step 3: Find Victim**
```
Set 0, hand = 0
Check slot 0:
  - bit = 0 (bits [00])
  - EVICT slot 0
  - hand -> 1
```

**Step 4: Insert**
```
Set 0 after insert:
Tags: [0x9A, 0x00, 0x00, 0x00]
Vals: [100,  -,    -,    -   ]
Clks: [11,   00,   00,   00  ]  (clock bits = 11 when inserted)
Hand: 1
```

### Insert Second Item

Command: `cache.upsert((Key(7), Value(200)))`

**Step 1: Hash**
```
hash = hash_builder.hash_one(7) = 0xXYZ9876
set_index = 0
tag = 0xBC
```

**Step 2: Search Set 0**
```
Load tags: [0x9A, 0x00, 0x00, 0x00]
SIMD compare with 0xBC: no matches
Miss!
```

**Step 3: Find Victim**
```
Set 0, hand = 1
Check slot 1:
  - bit = 0 (bits [00])
  - EVICT slot 1
  - hand -> 2
```

**Step 4: Insert**
```
Set 0 after insert:
Tags: [0x9A, 0xBC, 0x00, 0x00]
Vals: [100,  200,  -,    -   ]
Clks: [11,   11,   00,   00  ]
Hand: 2
```

### Hit on First Item

Command: `cache.get(&Key(5))`

**Step 1: Hash**
```
hash = hash_builder.hash_one(5) = 0xABCD1234
set_index = 0
tag = 0x9A
```

**Step 2: Load and Compare**
```
Load tags: [0x9A, 0xBC, 0x00, 0x00]
SIMD compare: [0xFF, 0x00, 0x00, 0x00]
Match at way = 0!
```

**Step 3: Verify Key**
```
Slot 0, value = 100
Stored key = 5
Key matches!
```

**Step 4: Update Clock**
```
slot = 0
clocks.set_bit(slot) -> increase clock counter

Set 0 after hit:
Clks: [11, 11, 00, 00]  (already at max, stays at 11)
```

**Return: Some(100)**

### Eviction Scenario

After many inserts, cache full:
```
Set 0:
Tags: [0x9A, 0xBC, 0xDE, 0xF0]
Clks: [00,   11,   10,   01  ]  (different ages)
Hand: 1
```

Command: Insert new item with tag 0xAA

**Step 1: Search - Miss**

**Step 2: Find Victim**
```
Set 0, hand = 1
Slot 1: bit = 11, clear -> 10, continue
Slot 2: bit = 10, clear -> 00, continue
Slot 3: bit = 01, clear -> 00, continue
Slot 0: bit = 00, EVICT!
Hand -> 1
```

**Step 3: Insert**
```
Set 0 after eviction:
Tags: [0xAA, 0xBC, 0xDE, 0xF0]  (slot 0 replaced)
Clks: [11,   10,   00,   00  ]  (clock bits cleared during scan)
Hand: 1
```

## Performance Analysis

### Time Complexity

| Operation | Complexity | Typical Cycles |
|-----------|-----------|-----------------|
| Hash | O(1) | 20 (pipelined) |
| Set/Tag extract | O(1) | 1-2 |
| SIMD compare | O(1) | 1-2 |
| Key verify | O(1) | 3-5 |
| Clock update | O(1) | 1 |
| CLOCK scan (amortized) | O(1) | 1-2 |
| **Total on hit** | **O(1)** | **~30 cycles** |

### Memory Access Pattern

```
Hit path:
1. Hash function uses SipHash (internal)
2. Load tags[set] <- L1 cache hit
3. SIMD compare <- register operation
4. Load value <- L1 cache hit (if same cache line)

Miss path:
Same as above, plus:
5. Find victim slot
6. Store new value <- L1 cache write
7. Update clock bit <- L1 cache hit
```

### Cache Efficiency

```
L1 cache access:
- Tags array: 16-32 bytes per set
- If set fits in L1 line (64B): single load
- Values: immediate follow tags
- Total: 2-3 memory operations for complete search

vs HashMap:
- Hash table lookup: 1-20 probes
- Chain traversal: 1-5 indirections
- Key comparison: variable cost
- Total: 5-10 memory operations

SetAssociative: 2-3 ops
HashMap: 5-10 ops

Advantage: 2.5-5x fewer memory operations
```

## Comparison with Other Algorithms

### SetAssociative vs LRU

LRU search (HashMap + VecDeque):
```
1. Hash function -> table index
2. Hash table probe (possibly chain traversal)
3. Get VecDeque position
4. Move to end: VecDeque::retain() -> O(n) scan and moves!
5. Total: O(1) search + O(n) reordering
```

SetAssociative search:
```
1. Hash function
2. SIMD tag match
3. Clock bit update
4. Total: O(1)
```

**Difference: LRU O(n) reordering vs SetAssoc O(1) bit flip**

### SetAssociative vs LFU

LFU search (HashMap + MinHeap):
```
1. Hash table lookup
2. Get frequency counter
3. Increment counter
4. MinHeap rebalance: O(log n)
5. Total: O(log n)
```

SetAssociative search:
```
1. SIMD tag match
2. Clock bit update
3. Total: O(1)
```

**Difference: LFU O(log n) heap vs SetAssoc O(1) bit flip**

## Conclusion

SetAssociativeCache achieves high performance through:

1. **Hardware-friendly layout**: Single cache line per set
2. **SIMD vectorization**: 16 tags compared in parallel
3. **O(1) operations**: No expensive data structure updates
4. **Compile-time optimization**: Constants and dead code elimination
5. **Predictable access**: No pointer chasing

Trade-off: Fixed size and approximate eviction (CLOCK vs exact LRU).

Result: 2-4x faster than standard policies on throughput.
