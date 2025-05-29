using CommunityToolkit.Mvvm.ComponentModel;
using Prism.Services.Dialogs;
using System;
using System.Windows.Interop;
using System.Windows.Media.Imaging;
using System.Windows.Media;
using System.Windows;
using System.Drawing;
using CommunityToolkit.Mvvm.Input;

namespace Cameca.CustomAnalysis.PeakDetection.ModelValidation;

internal partial class UntrustedModelViewModel : ObservableObject, IDialogAware
{
    public const string AddTrustedParamKey = "AddTrusted";

    public string? Path { get; set; }
    public string? Hash { get; set; }


    private static Icon warningIcon = SystemIcons.Warning;
    public static ImageSource ImageSource { get; } = Imaging.CreateBitmapSourceFromHIcon(
        warningIcon.Handle,
        Int32Rect.Empty,
        BitmapSizeOptions.FromEmptyOptions());



    public string Title { get; } = "Untrusted Model";

    public event Action<IDialogResult>? RequestClose;

    public bool CanCloseDialog() => true;

    public void OnDialogClosed() { }

    [RelayCommand]
    public void Abort() => RequestClose?.Invoke(new DialogResult(ButtonResult.No));

    [RelayCommand]
    public void TrustOnce() => RequestClose?.Invoke(new DialogResult(ButtonResult.Yes, new DialogParameters
    {
        { AddTrustedParamKey, false },
    }));

    [RelayCommand]
    public void AddTrusted() => RequestClose?.Invoke(new DialogResult(ButtonResult.Yes, new DialogParameters
    {
        { AddTrustedParamKey, true },
    }));

    public void OnDialogOpened(IDialogParameters parameters)
    {
        Path = parameters.GetValue<string>(nameof(Path));
        Hash = parameters.GetValue<string?>(nameof(Hash));
    }
}
