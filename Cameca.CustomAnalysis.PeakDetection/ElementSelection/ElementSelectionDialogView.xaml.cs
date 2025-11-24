using System;
using System.ComponentModel;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;

namespace Cameca.CustomAnalysis.PeakDetection.ElementSelection;

/// <summary>
/// Interaction logic for ElementSelectionDialogView.xaml
/// </summary>
public partial class ElementSelectionDialogView : UserControl
{
    public ElementSelectionDialogView()
    {
        InitializeComponent();

        var dpd = DependencyPropertyDescriptor.FromProperty(ItemsControl.ItemsSourceProperty, typeof(DataGrid));
        dpd.AddValueChanged(ModelsDataGrid, OnItemsSourceChanged);
    }

    private void OnItemsSourceChanged(object? sender, EventArgs e)
    {
        if (ModelsDataGrid.ItemsSource is not null)
        {
            if (CollectionViewSource.GetDefaultView(ModelsDataGrid.ItemsSource) is { } view)
            {
                using (view.DeferRefresh())
                {
                    view.SortDescriptions.Clear();
                    view.SortDescriptions.Add(new SortDescription("Element", ListSortDirection.Ascending));
                }
            }
            foreach (var col in ModelsDataGrid.Columns)
            {
                col.SortDirection = null;
            }
            if (ModelsDataGrid.Columns.Count > 0)
            {
                ModelsDataGrid.Columns[0].SortDirection = ListSortDirection.Ascending;
            }
        }
    }
}
