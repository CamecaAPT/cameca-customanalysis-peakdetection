using Cameca.CustomAnalysis.Interface;
using Cameca.CustomAnalysis.Utilities;
using System.Collections.Generic;
using System.Linq;

namespace Cameca.CustomAnalysis.PeakDetection;

internal static class OverlapResolver
{
    /// <summary>
    /// Very basic removal of range overlaps.
    /// Orders all ranges by minimum value.
    /// For each pair of ordered ranges, if the two ranges overlap, find the average of the lower max and higher min,
    /// then adjust the bounds lower and higher ranges to that average value.
    /// Update in place and continue.
    /// Overlaped ranges will result in two touching but not overlapping ranges at the average of their overlapping bounds.
    /// If higher range fully overlaps lower range (i.e. higher.Max is still smaller than lower.Max),
    /// no special handler is applied. This will results in a large truncation of the upper portion of the lower range
    /// so the results have no overlap.
    /// This approach is designed with the assumption that small overlaps are possible at boundaries,
    /// but large, full overlaps are unlikely.
    /// </summary>
    /// <param name="ranges"></param>
    /// <returns></returns>
    public static List<IonTypeInfoRange> RemoveOverlaps(IEnumerable<IonTypeInfoRange> ranges)
    {
        // Ensure resolved collection to avoid multiple enumeration
        var orderedRng = ranges.OrderBy(x => x.Min).ToList();

        // Ensure having more than 1 range
        if (!orderedRng.Skip(1).Any())
        {
            return orderedRng;
        }

        // Guarenteed to have 2+ elements
        for (int i = 1; i < orderedRng.Count; i++)
        {
            var r1 = orderedRng[i-1];
            var r2 = orderedRng[i];

            if (RangeUtils.RangesOverlap(r1, r2))
            {
                double avg = new[] { r1.Max, r2.Min }.Average();
                var newR1 = new IonTypeInfoRange(
                    r1.Name,
                    r1.Formula,
                    r1.Volume,
                    r1.Min,
                    avg,
                    r1.Color);
                var newR2 = new IonTypeInfoRange(
                    r2.Name,
                    r2.Formula,
                    r2.Volume,
                    avg,
                    r2.Max,
                    r2.Color);
                orderedRng[i - 1] = newR1;
                orderedRng[i] = newR2;
            }
        }

        return orderedRng;
    }
}
