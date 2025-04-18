using Cameca.CustomAnalysis.Interface;
using Cameca.CustomAnalysis.Utilities;
using System.Linq;

namespace Cameca.CustomAnalysis.PeakDetection;

internal static class ResourceExtensions
{
    public static IonTypeInfoRange CreateIonTypeInfoRange(this IResources resources, string name, double lower, double upper)
    {
        var formula = IonFormulaEx.Parse(name);
        double volume = resources.ElementData is not null ? CalculateVolume(resources.ElementData, formula) : 0d;
        var color = resources.GetIonColor(name, formula);
        return new IonTypeInfoRange(name, formula, volume, lower, upper, color);
    }

    /// <summary>
    /// vol/mol in cm^3 to vol/atom in nm^3
    /// (1.0e7)^3 / 6.022e23
    /// </summary>
    public const double AtomicColConv = 0.0016606d;

    private static double CalculateVolume(IElementDataSet elemenData, IonFormula formula)
    {
        double totalVolume = 0d;
        foreach (var (symbol, count) in formula)
        {
            var volume = elemenData.Elements.FirstOrDefault(x => x.Symbol == symbol)?.MolarVolume ?? 0;
            totalVolume = volume * count;
        }
        return totalVolume * AtomicColConv;
    }
}