$ErrorActionPreference = 'Stop'

$previewConfig = Join-Path ([IO.Path]::GetTempPath()) ("blog-preview-{0}.toml" -f [guid]::NewGuid().ToString('N'))

Push-Location -LiteralPath $PSScriptRoot
try {
    @'
[[module.mounts]]
source = "content"
target = "content"

[[module.mounts]]
source = "no_render"
target = "content/posts"
'@ | Set-Content -LiteralPath $previewConfig -Encoding utf8

    Write-Host 'Preview: http://localhost:1313/zh/posts/ (Ctrl+C to stop)'
    & hugo server --config "config.toml,$previewConfig" --buildDrafts --buildFuture --renderToMemory
    if ($LASTEXITCODE -ne 0) {
        throw "Hugo preview failed with exit code $LASTEXITCODE."
    }
}
finally {
    Remove-Item -LiteralPath $previewConfig -Force
    Pop-Location
}
