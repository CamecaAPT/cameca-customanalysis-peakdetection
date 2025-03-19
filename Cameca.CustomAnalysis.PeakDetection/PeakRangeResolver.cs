using Cameca.CustomAnalysis.Interface;
using Cameca.CustomAnalysis.Utilities;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Cameca.CustomAnalysis.PeakDetection;

internal static class PeakRangeResolver
{
    public static IEnumerable<IonTypeInfoRange> FromExistingRanges(IEnumerable<IonTypeInfoRange> newPeaks, IEnumerable<IonTypeInfoRange> existingRanges)
    {
        ICollection<IonTypeInfoRange> matchRanges = existingRanges is ICollection<IonTypeInfoRange> x ? x : existingRanges.ToList();
        foreach (var newPeak in newPeaks)
        {
            var overlapping = matchRanges.Where(r => RangeUtils.RangesOverlap(newPeak, r)).ToList();

            if (overlapping.Any())
            {
                var match = overlapping.Count == 1
                    ? overlapping.Single()
                    : GetLargestOverlap(newPeak, overlapping);
                yield return new IonTypeInfoRange(match.Name, match.Formula, match.Volume, newPeak.Min, newPeak.Max, match.Color);
            }
            else
            {
                yield return newPeak;
            }
        }
    }

    private static IonTypeInfoRange GetLargestOverlap(IonTypeInfoRange target, IEnumerable<IonTypeInfoRange> matches)
    {
        IonTypeInfoRange? largestOverlap = null;
        double maxOverlap = double.MinValue;
        foreach (var match in matches)
        {
            var overlap = CalculateOverlap(target, match);
            if (overlap > maxOverlap)
            {
                maxOverlap = overlap;
                largestOverlap = match;
            }
        }
        return largestOverlap ?? throw new ArgumentException("Requires at least one overlapping match", nameof(matches));
    }

    private static double CalculateOverlap(IonTypeInfoRange a, IonTypeInfoRange b)
    {
        return Math.Min(a.Max, b.Max) - Math.Max(a.Min, b.Min);
    }
}
