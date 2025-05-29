using Cameca.CustomAnalysis.Interface;
using System.ComponentModel.DataAnnotations;

namespace Cameca.CustomAnalysis.PeakDetection;

public partial class GlobalPeakDetectionProperties
{
    [Display(Name = "Default Method")]
    public PeakDetectionImplementation DefaultMethod { get; set; } = PeakDetectionImplementation.RandomForest;

    public string? defaultModelPath = null;
    [Display(Name = "Default Model Path")]
    [FilePath(AllowMultiple = false, Filter = "Model File (Python Pickle file) (*.pkl)|*.pkl|All Files (*.*)|*.*")]
    public string? DefaultModelPath
    {
        get => defaultModelPath;
        set
        {
            try
            {
                defaultModelPath = PathTokenizer.Tokenize(value);
            }
            catch
            {
                // Don't want to fail on setting if something wrong with EnvVar tokenization
                // Fallback to just setting the value
                defaultModelPath = value;
            }
        }
    }

    public string? defaultScalarPath = null;
    [Display(Name = "Default Scalar Path")]
    [FilePath(AllowMultiple = false, Filter = "Scalar File (Python Pickle file) (*.pkl)|*.pkl|All Files (*.*)|*.*")]
    public string? DefaultScalarPath
    {
        get => defaultScalarPath;
        set
        {
            try
            {
                defaultScalarPath = PathTokenizer.Tokenize(value);
            }
            catch
            {
                // Don't want to fail on setting if something wrong with EnvVar tokenization
                // Fallback to just setting the value
                defaultScalarPath = value;
            }
        }
    }

    [Display(Description = "Object confidence threshold for detection")]
    public double Confidence { get; set; } =  0.2d;
    
    [Display(Name = "Intersection Over Union", Description = "Intersection over union (IoU) threshold for NMS: higher means get more")]
    public double IntersectionOverUnion { get; set; } = 0.01d;

    [Display(Name = "Max Detections", Description = "Maximum number of detections per image")]
    public int MaxDetections { get; set; } = 2000;
}
