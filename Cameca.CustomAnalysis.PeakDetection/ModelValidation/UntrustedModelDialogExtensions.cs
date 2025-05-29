using Prism.Services.Dialogs;

namespace Cameca.CustomAnalysis.PeakDetection.ModelValidation;

internal static class UntrustedModelDialogExtensions
{
    public static UntrustedModelDialogResult ShowUntrustedModelDialog(this IDialogService dialogService, string path, string? hash)
    {
        IDialogResult? dialogResult = null;
        dialogService.ShowDialog(
            nameof(UntrustedModelViewModel),
            new DialogParameters
            {
                { nameof(UntrustedModelViewModel.Path), path },
                { nameof(UntrustedModelViewModel.Hash), hash },
            },
            // Capture results
            (results) => dialogResult = results);
        if (dialogResult is not null && dialogResult.Result == ButtonResult.Yes)
        {
            bool addTrusted = dialogResult.Parameters.GetValue<bool>(UntrustedModelViewModel.AddTrustedParamKey);
            return new UntrustedModelDialogResult(true, addTrusted);
        }
        return new UntrustedModelDialogResult(false, false);
    }
}
