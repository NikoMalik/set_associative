//! Allocator abstraction layer.
//!
//! With the default `allocator-api2` feature, re-exports from `allocator_api2`.
//! Without it, provides a minimal `Allocator` trait backed by the global allocator.

use core::ptr::NonNull;
use std::alloc::Layout;

#[cfg(feature = "allocator-api2")]
pub use allocator_api2::alloc::{AllocError, Allocator, Global};

#[cfg(not(feature = "allocator-api2"))]
pub unsafe trait Allocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError>;
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout);
}

#[cfg(not(feature = "allocator-api2"))]
#[derive(Debug, Clone, Copy)]
pub struct AllocError;

#[cfg(not(feature = "allocator-api2"))]
impl core::fmt::Display for AllocError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "memory allocation failed")
    }
}

#[cfg(not(feature = "allocator-api2"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct Global;

#[cfg(not(feature = "allocator-api2"))]
unsafe impl Allocator for Global {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.size() == 0 {
            // SAFETY: alignment is non-zero
            let ptr = unsafe { NonNull::new_unchecked(layout.align() as *mut u8) };
            return Ok(NonNull::slice_from_raw_parts(ptr, 0));
        }
        // SAFETY: layout has non-zero size
        let ptr = unsafe { std::alloc::alloc(layout) };
        let ptr = NonNull::new(ptr).ok_or(AllocError)?;
        Ok(NonNull::slice_from_raw_parts(ptr, layout.size()))
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() == 0 {
            return;
        }
        // SAFETY: caller guarantees ptr was allocated with this layout
        unsafe { std::alloc::dealloc(ptr.as_ptr(), layout) }
    }
}

/// Helper to allocate via an `Allocator`, returning `NonNull<[u8]>`.
pub fn do_alloc<A: Allocator>(alloc: &A, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
    alloc.allocate(layout)
}
