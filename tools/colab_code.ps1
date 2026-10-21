# Chi hoi ma dang nhap (cua lan mo trang Google gan nhat) va luu token. Neu ma sai / het han thi chuyen sang colab_login.ps1.
$py = "$env:APPDATA\uv\tools\google-colab-cli\Scripts\python.exe"
$token = "$env:USERPROFILE\.config\colab-cli\token.json"
Set-Location (Split-Path $PSScriptRoot)
$before = if (Test-Path $token) { (Get-Item $token).LastWriteTime } else { $null }
Write-Host ""
$code = Read-Host "Dan ma 4/0... vao day (chuot phai de dan) roi bam Enter"
$code.Trim() | & $py tools\colab_login.py exchange
$after = if (Test-Path $token) { (Get-Item $token).LastWriteTime } else { $null }
if ($after -and $after -ne $before) {
    Write-Host ""
    Write-Host "==== THANH CONG: da luu token. Bao Claude 'xong'. ====" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "==== Ma nay het han. Trang Google se mo lai: bam Allow, Copy ma moi, dan vao day. ====" -ForegroundColor Red
    & "$PSScriptRoot\colab_login.ps1"
}
