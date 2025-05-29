using System;
using System.Collections;
using System.IO;

namespace Cameca.CustomAnalysis.PeakDetection;

internal static class PathTokenizer
{
    public static string? Tokenize(string? path)
    {
        if (path is null)
        {
            return path;
        }
        var fullpath = Path.GetFullPath(new Uri(Environment.ExpandEnvironmentVariables(path)).LocalPath)
            .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);

        string? matchKey = null;
        string? matchValue = null;
        int longestMatchLength = 0;
        foreach (DictionaryEntry envEntry in Environment.GetEnvironmentVariables())
        {
            if (envEntry is { Key: string key, Value: string value })
            {
                if (value.Length > longestMatchLength && fullpath.StartsWith(value, StringComparison.OrdinalIgnoreCase))
                {
                    longestMatchLength = value.Length;
                    matchKey = key;
                    matchValue = value;
                }
            }
        }
        if (matchKey is not null && matchValue is not null)
        {
            return Path.Join($"%{matchKey}%", fullpath.Substring(matchValue.Length));
        }

        return path;
    }
}
