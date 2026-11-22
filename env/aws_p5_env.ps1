<#
.SYNOPSIS
  Puts one P5 IAM user's AWS keys into THIS PowerShell session's environment, and nothing
  anywhere else (user rules, 2026-11-22).

.DESCRIPTION
  Run it in the window that will use the keys (environment variables are per process; a
  script runs inside the calling PowerShell process, so they stay set after it returns):

      Set-ExecutionPolicy -Scope Process Bypass     # this window only, if scripts are blocked
      .\env\aws_p5_env.ps1 -Role agent              # or: -Role launch
      .\env\aws_p5_env.ps1 -Clear                   # removes them again

  - Both values are read with Read-Host -AsSecureString: nothing is echoed, and Read-Host
    answers never enter the PSReadLine history (only this command line does, and it holds
    no secret).
  - It sets AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY and AWS_DEFAULT_REGION=us-east-2 for
    this process only. It never writes a file: no ~/.aws, no profile, no transcript.
  - AWS_CONFIG_FILE and AWS_SHARED_CREDENTIALS_FILE point at a path that does not exist, so
    the AWS CLI can neither read stray keys from ~/.aws nor be pointed at a profile.
  - It checks the keys with `aws sts get-caller-identity` and refuses (clearing them) unless
    the IAM user is dbp-<Role>. Only the user name is printed, never the keys or the
    account id.
#>
[CmdletBinding(DefaultParameterSetName = 'Set')]
param(
    [Parameter(Mandatory = $true, ParameterSetName = 'Set')]
    [ValidateSet('agent', 'launch')]
    [string]$Role,

    [Parameter(Mandatory = $true, ParameterSetName = 'Clear')]
    [switch]$Clear
)

$names = @('AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN', 'AWS_PROFILE',
           'AWS_DEFAULT_PROFILE', 'DBP_AWS_ROLE')

function Remove-DbpAwsKeys {
    foreach ($name in $names) {
        if (Test-Path "Env:$name") { Remove-Item "Env:$name" }
    }
}

function Read-DbpSecret([string]$Prompt) {
    $secure = Read-Host -Prompt $Prompt -AsSecureString
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try { return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr) }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
}

if ($Clear) {
    Remove-DbpAwsKeys
    Write-Host 'AWS keys removed from this PowerShell session.'
    return
}

if (-not (Get-Command aws -ErrorAction SilentlyContinue)) {
    throw 'aws (AWS CLI v2) is not on PATH: install AWSCLIV2.msi, then open a new window.'
}

$noFiles = Join-Path $env:TEMP 'dbp-aws-no-files-here'
if (Test-Path $noFiles) {
    throw "$noFiles exists; it must not (the AWS CLI is pointed at it so it reads no file)."
}

Remove-DbpAwsKeys
$keyId = (Read-DbpSecret "dbp-$Role access key id").Trim()
$secret = (Read-DbpSecret "dbp-$Role secret access key").Trim()
if ($keyId -notmatch '^[A-Z0-9]{16,128}$' -or $secret.Length -lt 20) {
    $keyId = $null; $secret = $null
    throw 'That does not look like an access key id and secret; nothing was set.'
}

$env:AWS_ACCESS_KEY_ID = $keyId
$env:AWS_SECRET_ACCESS_KEY = $secret
$keyId = $null; $secret = $null
$env:AWS_DEFAULT_REGION = 'us-east-2'
$env:AWS_REGION = 'us-east-2'
# Regional STS, so get-caller-identity is a us-east-2 request: dbp-launch denies every
# request whose aws:RequestedRegion is not us-east-2, and the global STS endpoint is us-east-1.
$env:AWS_STS_REGIONAL_ENDPOINTS = 'regional'
$env:AWS_CONFIG_FILE = Join-Path $noFiles 'config'
$env:AWS_SHARED_CREDENTIALS_FILE = Join-Path $noFiles 'credentials'
$env:AWS_PAGER = ''
$env:AWS_CLI_AUTO_PROMPT = 'off'

$arn = aws sts get-caller-identity --query Arn --output text
if ($LASTEXITCODE -ne 0 -or -not $arn) {
    Remove-DbpAwsKeys
    throw 'aws sts get-caller-identity failed with these keys; they were removed again.'
}
$user = ($arn.Trim() -split '/')[-1]
if ($user -ne "dbp-$Role") {
    Remove-DbpAwsKeys
    throw "These keys belong to IAM user '$user', not 'dbp-$Role'; they were removed again."
}
$env:DBP_AWS_ROLE = $Role
Write-Host "AWS keys set for this PowerShell session only: IAM user $user, region us-east-2."
Write-Host 'Nothing was written to disk. Remove them with: .\env\aws_p5_env.ps1 -Clear'
if (Test-Path (Join-Path $HOME '.aws')) {
    Write-Host "Note: $HOME\.aws exists; this session ignores it (AWS_CONFIG_FILE points elsewhere)."
}
