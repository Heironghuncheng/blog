$ErrorActionPreference = 'Stop'

$previewConfig = Join-Path ([IO.Path]::GetTempPath()) ("blog-preview-{0}.json" -f [guid]::NewGuid().ToString('N'))

Push-Location -LiteralPath $PSScriptRoot
try {
    $configJson = & hugo config --config config.toml --format json
    if ($LASTEXITCODE -ne 0) {
        throw "Hugo config failed with exit code $LASTEXITCODE."
    }
    $config = $configJson -join "`n" | ConvertFrom-Json
    # Hugo replaces mount arrays when loading an override.
    $mounts = @($config.module.mounts) + @{ source = 'no_render'; target = 'content/posts' }
    @{ module = @{ mounts = $mounts } } | ConvertTo-Json -Depth 5 |
        Set-Content -LiteralPath $previewConfig -Encoding utf8

    Write-Host 'Preview: http://localhost:1313/zh/posts/ (Ctrl+C to stop)'
    & hugo server --config "config.toml,$previewConfig" --buildDrafts --buildFuture --renderToMemory
    if ($LASTEXITCODE -ne 0) {
        throw "Hugo preview failed with exit code $LASTEXITCODE."
    }
}
finally {
    if (Test-Path -LiteralPath $previewConfig) {
        Remove-Item -LiteralPath $previewConfig -Force
    }
    Pop-Location
}
