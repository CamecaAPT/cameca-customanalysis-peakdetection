using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;

namespace Cameca.CustomAnalysis.PeakDetection.ModelValidation;

internal record ModelValidatorResult(bool IsTrusted, string? Hash);

internal class ModelValidator
{
    private readonly ILogger<ModelValidator> logger;
    private readonly char[] HexCharacters = {
        '0', '1', '2', '3', '4', '5', '6', '7',
        '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'
    };
    private const string TrustedModelsFilePath = "TrustedModels.txt";

    public ModelValidator(ILogger<ModelValidator> logger)
    {
        this.logger = logger;
    }

    public ModelValidatorResult ValidateModelFile(string path)
    {
        var expandedPath = Environment.ExpandEnvironmentVariables(path);
        var trustedHashes = GetTrustedHashes();

        string? hash = null;
        using (var stream = new FileStream(expandedPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
        {
            hash = ComputeHash(stream);
        }
        bool isTrusted = trustedHashes.Contains(hash, StringComparer.OrdinalIgnoreCase);
        return new ModelValidatorResult(isTrusted, hash);
    }

    public void AddTrustedHash(string hash)
    {
        var normalizedhash = NormalizeHash(hash);
        if (IsValidSha256HexString(normalizedhash))
        {
            var fileInfo = TrustedModelsFileInfo();
            using var stream = fileInfo.Open(FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.Read);
            stream.Position = 0;
            var existingLines = ReadAllLines(stream).ToList();
            if (!existingLines.Contains(normalizedhash))
            {
                using var writer = new StreamWriter(stream, leaveOpen: true);
                writer.WriteLine(normalizedhash);
            }
        }
    }

    private IEnumerable<string> ReadAllLines(Stream stream)
    {
        using var reader = new StreamReader(stream, leaveOpen: true);
        string? line = reader.ReadLine();
        while (line is not null)
        {
            yield return line;
            line = reader.ReadLine();
        }
    }

    private FileInfo TrustedModelsFileInfo()
    {
        if (Assembly.GetExecutingAssembly().Location is string dllPath
            && !string.IsNullOrWhiteSpace(dllPath)
            && new FileInfo(dllPath).DirectoryName is string directory)
        {
            return new FileInfo(Path.Join(directory, TrustedModelsFilePath));
        }
        else
        {
            return new FileInfo(TrustedModelsFilePath);
        }

    }

    private string NormalizeHash(string hash)
    {
        return hash.Replace("-", "").Trim().ToLowerInvariant();
    }

    private List<string> GetTrustedHashes()
    {
        try
        {
            var fileInfo = TrustedModelsFileInfo();
            using var stream = fileInfo.Open(FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
            stream.Position = 0;
            var hashes = new List<string>();
            foreach (var line in ReadAllLines(stream))
            {
                var normalizedLine = NormalizeHash(line);
                if (IsValidSha256HexString(normalizedLine))
                {
                    hashes.Add(normalizedLine);
                }
            }
            return hashes;
        }
        catch (Exception e)
        {
            logger.LogWarning(e, "Could not retrieve a list of trusted model hashes");
            return new List<string>();
        }
    }

    private bool IsValidSha256HexString(string? value)
    {
        if (value is null) return false;
        if (value.Length != 64) return false;
        return value.ToLowerInvariant().All(x => HexCharacters.Contains(x));
    }

    private string ComputeHash(FileStream stream)
    {
        using var sha256 = SHA256.Create();
        stream.Position = 0;
        return NormalizeHash(BitConverter.ToString(sha256.ComputeHash(stream)));
    }
}
