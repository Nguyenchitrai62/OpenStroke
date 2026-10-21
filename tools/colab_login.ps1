# Dang nhap Colab CLI: mo trang Google, hoi ma, luu token. Chay bang colab_login.bat o thu muc goc repo.
$py = "$env:APPDATA\uv\tools\google-colab-cli\Scripts\python.exe"
$token = "$env:USERPROFILE\.config\colab-cli\token.json"
Set-Location (Split-Path $PSScriptRoot)
while ($true) {
    $before = if (Test-Path $token) { (Get-Item $token).LastWriteTime } else { $null }
    $url = & $py tools\colab_login.py url
    Start-Process $url
    Write-Host ""
    Write-Host "Trinh duyet vua mo trang dang nhap Google."
    Write-Host "Chon tai khoan -> bam Allow / Cho phep -> bam Copy de copy ma."
    Write-Host ""
    $code = Read-Host "Quay lai cua so nay, bam chuot phai de dan ma, roi bam Enter"
    $code.Trim() | & $py tools\colab_login.py exchange
    $after = if (Test-Path $token) { (Get-Item $token).LastWriteTime } else { $null }
    if ($after -and $after -ne $before) {
        Write-Host ""
        Write-Host "==== THANH CONG: da luu token. Dong cua so nay va bao Claude 'xong'. ====" -ForegroundColor Green
        break
    }
    Write-Host ""
    Write-Host "==== CHUA DUOC (ma sai hoac het han). Thu lai: trang Google se mo lai. ====" -ForegroundColor Red
}
