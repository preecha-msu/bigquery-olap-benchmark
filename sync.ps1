param (
    [string]$Message = "Sync LaTeX files: $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
)

Set-Location $PSScriptRoot

$remoteBranch = git ls-remote --heads origin main
if ($LASTEXITCODE -ne 0) {
    Write-Host "Cannot connect to GitHub! Please verify account permission or repo name." -ForegroundColor Red
    exit 1
}

if (-not [string]::IsNullOrWhiteSpace($remoteBranch)) {
    Write-Host "1. Pulling latest files from GitHub..." -ForegroundColor Cyan
    git pull origin main --rebase
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Pull failed! Please check for file conflicts." -ForegroundColor Red
        exit 1
    }
}

Write-Host "2. Checking modified files..." -ForegroundColor Cyan
git add .
if ($LASTEXITCODE -ne 0) {
    Write-Host "git add failed! Check if there is another nested .git folder." -ForegroundColor Red
    exit 1
}

if ([string]::IsNullOrWhiteSpace((git status --porcelain))) {
    Write-Host "Everything is up to date!" -ForegroundColor Green
} else {
    Write-Host "3. Uploading to GitHub..." -ForegroundColor Cyan
    git commit -m "$Message"
    git push -u origin main
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Sync completed!" -ForegroundColor Green
    } else {
        Write-Host "Push failed! Files were committed locally but not uploaded." -ForegroundColor Red
        exit 1
    }
}
