# RF Filter Implementation Summary

## Overview
This document summarizes the changes made to `pareto.py` to ensure that only recipes with `rf2 > rf1` are proposed and added to Excel files.

## Changes Made

### 1. Main Recipe Generation Function (`_build_and_write_proposals`)
- **Location**: Lines ~1080-1300
- **Changes**: Added rf2 > rf1 filter after Pareto improvement filtering
- **Logic**: 
  - First filters candidates by Pareto improvement criteria
  - Then applies rf2 > rf1 constraint
  - Falls back to relaxed criteria if no candidates meet both requirements
  - Maintains rf2 > rf1 constraint even in fallback scenarios

### 2. Iteration-Based Recipe Generation (`_propose_next_iteration_recipes`)
- **Location**: Lines ~2250-2350
- **Changes**: Added rf2 > rf1 filter after target rate filtering
- **Logic**:
  - Applies target rate filter first
  - Then applies rf2 > rf1 filter
  - Combines both filters for final candidate selection
  - Provides detailed logging of filter results

### 3. Regeneration Function (`_generate_fresh_proposals_for_iteration`)
- **Location**: Lines ~4900-5018
- **Changes**: Added rf2 > rf1 filter in two places:
  - After target rate filtering (similar to iteration function)
  - In the candidate selection loop (similar to main function)
- **Logic**: Ensures consistency across all recipe generation paths

### 4. Documentation and Logging
- **Location**: Top of file and throughout functions
- **Changes**: 
  - Added comment explaining the constraint
  - Enhanced debug logging to show rf1/rf2 values
  - Added filter statistics and warnings

## Filter Logic

The rf2 > rf1 filter works as follows:

```python
# Apply rf2 > rf1 filter
rf_filtered_candidates = []
for i in valid_candidates:
    rf1_power = X_band[i, FEATURES.index("Etch_Avg_Rf1_Pow")]
    rf2_power = X_band[i, FEATURES.index("Etch_Avg_Rf2_Pow")]
    if rf2_power > rf1_power:
        rf_filtered_candidates.append(i)
    else:
        print(f"Filtered out candidate {i}: Rf1={rf1_power:.1f}, Rf2={rf2_power:.1f} (rf2 must be > rf1)")
```

## Fallback Strategy

If no candidates meet both Pareto improvement and rf2 > rf1 requirements:

1. **First fallback**: Try with relaxed Pareto thresholds (50% of original)
2. **Second fallback**: Use original non-dominated criteria
3. **Final fallback**: If still no candidates, stop and warn user

The rf2 > rf1 constraint is maintained throughout all fallback scenarios.

## Impact

- **Before**: All Pareto-improving recipes were proposed regardless of rf1/rf2 relationship
- **After**: Only recipes with rf2 > rf1 are proposed
- **Benefit**: Ensures experimental recipes follow the expected power hierarchy
- **Safety**: Prevents invalid recipe configurations from being proposed

## Testing

The filter logic was tested with various rf1/rf2 combinations:
- ✓ Valid: rf2 > rf1 (e.g., Rf1=30, Rf2=80)
- ✗ Invalid: rf2 ≤ rf1 (e.g., Rf1=80, Rf2=30 or Rf1=50, Rf2=50)

## Files Modified

- `pareto.py` - Main implementation file
- All changes are backward compatible and don't affect existing functionality

## Notes

- The filter is applied at multiple levels to ensure consistency
- Detailed logging helps debug any filtering issues
- Fallback strategies ensure the system remains robust
- The constraint is enforced across all recipe generation paths
