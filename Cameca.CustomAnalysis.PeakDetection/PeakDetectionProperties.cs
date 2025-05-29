using Cameca.CustomAnalysis.Interface;
using CommunityToolkit.Mvvm.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Numerics;

namespace Cameca.CustomAnalysis.PeakDetection;

public partial class PeakDetectionProperties : ObservableObject
{
    [ObservableProperty]
    [field: Display(Name = "Method", Description = "Ensure the method matches that of the selected model file")]
    private PeakDetectionImplementation implementation = PeakDetectionImplementation.RandomForest;

    private string? modelPath = null;
    [Display(Name = "Model Path")]
    [FilePath(AllowMultiple = false, Filter = "Model File (Python Pickle file) (*.pkl)|*.pkl|All Files (*.*)|*.*")]
    public string? ModelPath
    {
        get => modelPath;
        set => SetProperty(ref modelPath, value);
    }

    private string? scalarPath = null;
    [Display(Name = "Scalar Path")]
    [FilePath(AllowMultiple = false, Filter = "Scalar File (Python Pickle file) (*.pkl)|*.pkl|All Files (*.*)|*.*")]
    public string? ScalarPath
    {
        get => scalarPath;
        set => SetProperty(ref scalarPath, value);
    }

    [ObservableProperty]
    [field: Display(Description = "Object confidence threshold for detection")]
    private double confidence = 0.2d;

    [ObservableProperty]
    [field: Display(Name = "Intersection Over Union", Description = "Intersection over union (IoU) threshold for NMS: higher means get more")]
    private double intersectionOverUnion = 0.01d;

    [ObservableProperty]
    [field: Display(Name = "Max Detections", Description = "Maximum number of detections per image")]
    private int maxDetections = 2000;

    [ObservableProperty]
    [field: Display(AutoGenerateField = false)]
    private Vector2 viewportLower;

    [ObservableProperty]
    [field: Display(AutoGenerateField = false)]
    private Vector2 viewportUpper;
}

public enum PeakDetectionImplementation
{
    [Display(Name = "Neural Network")]
    NeuralNetwork = 1,
    [Display(Name = "Random Forest")]
    RandomForest = 2,
}