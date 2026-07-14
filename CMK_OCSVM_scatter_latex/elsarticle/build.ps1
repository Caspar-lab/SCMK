# build.ps1 — 一键编译 elsarticle LaTeX
# 用法（在本目录）:  powershell -ExecutionPolicy Bypass -File build.ps1
#        编译其他文件:  powershell -ExecutionPolicy Bypass -File build.ps1 method
# 自动跑 3 遍 pdflatex 以解析交叉引用与参考文献，并检查未解析引用。
param([string]$Name = 'manuscript')

$pdflatex = 'C:\Users\shihao\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe'
if (-not (Test-Path $pdflatex)) { Write-Host "找不到 pdflatex: $pdflatex"; exit 1 }

Set-Location $PSScriptRoot
$tex = "$Name.tex"
if (-not (Test-Path $tex)) { Write-Host "找不到 $tex"; exit 1 }

Write-Host "Compiling $tex (3 passes)..."
for ($i = 1; $i -le 3; $i++) {
    Write-Host "  pass $i ..."
    & $pdflatex -interaction=nonstopmode $tex *> "$Name.build.log"
}

if (Test-Path "$Name.pdf") {
    $pg = (Select-String -Path "$Name.log" -Pattern 'Output written on .*\((\d+) page').Matches.Value
    Write-Host "OK: $Name.pdf  ($pg)"
    $warn = Select-String -Path "$Name.log" `
        -Pattern 'Undefined|Citation.*undefined|Reference.*undefined|Emergency stop'
    if ($warn) { Write-Host "未解析引用/错误:"; $warn | Select-Object -First 10 | ForEach-Object { Write-Host "  $($_.Line)" } }
    else { Write-Host "无未解析引用" }
    Remove-Item "$Name.build.log" -ErrorAction SilentlyContinue
} else {
    Write-Host "FAILED：未生成 PDF，详见 $Name.log"
    exit 1
}
