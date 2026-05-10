using Cameca.CustomAnalysis.PythonCore;
using PolyType;
using StreamJsonRpc;
using System;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Documents;

namespace Cameca.CustomAnalysis.PeakDetection;

public sealed record PredictRangesResult(float[][] PeakPred, string[] Elem1, float[] Conf1, string[] Elem2, float[] Conf2, int[] PeakIter);

[JsonRpcContract]
[GenerateShape(IncludeMethods = MethodShapeFlags.PublicInstance)]
public partial interface IPythonApi : IPythonApiBase
{
    /// <summary>
    /// Eagerly ensures training data is loaded and cached.
    /// </summary>
    /// <remarks>
    /// All Python methods that need it will load an wait,
    /// but a separate method can be used to show a progress
    /// bar that otherwise isn't displayed.
    /// </remarks>
    /// <returns></returns>
    [JsonRpcMethod("preload_training_data")]
    Task PreloadTrainingData(IProgress<double> progress, CancellationToken cancellationToken);

    [JsonRpcMethod("predict_ranges")]
    Task<PredictRangesResult> PredictRanges(
        string[] elementList,
        string[] elementsOtMolecules,
        double confidence,
        double intersectionOverUnion,
        int maxDetections,
        int iterations);

    [JsonRpcMethod("recommend_elements")]
    Task<string[]> RecommendElements(
        double[][] peakRangePred,
        string[] elements,
        double threshold,
        int numElements);
}