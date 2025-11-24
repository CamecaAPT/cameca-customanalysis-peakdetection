using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Data;

namespace Cameca.CustomAnalysis.PeakDetection.ElementSelection;

internal class ElementSelectionModelConverter : IValueConverter
{
    public bool FilterToIncludeComplex { get; set; } = false;

    public object Convert(object value, Type targetType, object parameter, CultureInfo culture)
    {
        if (value is IEnumerable<ElementSelectionModel> elements)
        {
            var filteredElements = elements
                .Where(x => !FilterToIncludeComplex || x.IncludeComplex)
                .Select(x => x.Element)
                .OrderBy(x => x);
            return string.Join(", ", filteredElements);
        }
        return DependencyProperty.UnsetValue;
    }

    public object ConvertBack(object value, Type targetType, object parameter, CultureInfo culture)
    {
        throw new NotImplementedException();
    }
}
