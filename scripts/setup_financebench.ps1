[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ZipPath
)

$ErrorActionPreference = "Stop"

$resolvedZip = (Resolve-Path -LiteralPath $ZipPath).Path
if ([System.IO.Path]::GetExtension($resolvedZip) -ne ".zip") {
    throw "Expected a .zip file: $resolvedZip"
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$destination = Join-Path $projectRoot "data\benchmarks\financebench"
$pdfDestination = Join-Path $destination "pdfs"
$temporaryRoot = Join-Path ([System.IO.Path]::GetTempPath()) (
    "financebench-" + [System.Guid]::NewGuid().ToString("N")
)

try {
    New-Item -ItemType Directory -Path $temporaryRoot | Out-Null
    Expand-Archive -LiteralPath $resolvedZip -DestinationPath $temporaryRoot

    $questionsSource = Get-ChildItem -Path $temporaryRoot -Recurse -File `
        -Filter "financebench_open_source.jsonl" |
        Select-Object -First 1

    $metadataSource = Get-ChildItem -Path $temporaryRoot -Recurse -File `
        -Filter "financebench_document_information.jsonl" |
        Select-Object -First 1

    $pdfSources = @(
        Get-ChildItem -Path $temporaryRoot -Recurse -File -Filter "*.pdf"
    )

    if ($null -eq $questionsSource) {
        throw "financebench_open_source.jsonl was not found in the archive."
    }
    if ($null -eq $metadataSource) {
        throw "financebench_document_information.jsonl was not found in the archive."
    }
    if ($pdfSources.Count -eq 0) {
        throw "No PDF files were found in the archive."
    }

    New-Item -ItemType Directory -Force -Path $pdfDestination | Out-Null

    Copy-Item -LiteralPath $questionsSource.FullName `
        -Destination (Join-Path $destination $questionsSource.Name) `
        -Force

    Copy-Item -LiteralPath $metadataSource.FullName `
        -Destination (Join-Path $destination $metadataSource.Name) `
        -Force

    foreach ($pdf in $pdfSources) {
        Copy-Item -LiteralPath $pdf.FullName `
            -Destination (Join-Path $pdfDestination $pdf.Name) `
            -Force
    }

    $questionsPath = Join-Path $destination "financebench_open_source.jsonl"
    $questionCount = (Get-Content -LiteralPath $questionsPath | Measure-Object -Line).Lines
    $copiedPdfCount = @(
        Get-ChildItem -LiteralPath $pdfDestination -File -Filter "*.pdf"
    ).Count

    if ($questionCount -ne 150) {
        Write-Warning "Expected 150 FinanceBench questions, found $questionCount."
    }

    Write-Host "FinanceBench setup completed."
    Write-Host "Questions: $questionCount"
    Write-Host "PDF files: $copiedPdfCount"
    Write-Host "Destination: $destination"
}
finally {
    if (Test-Path -LiteralPath $temporaryRoot) {
        Remove-Item -LiteralPath $temporaryRoot -Recurse -Force
    }
}
