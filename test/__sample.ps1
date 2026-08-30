# diagnose_wifi.ps1 - Captures Intel 9560 and MEI state
$logFile = "$env:TEMP\wifi_diag_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

function Write-Log($msg) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$timestamp | $msg" | Tee-Object -FilePath $logFile -Append
}

Write-Log "=== WIFI DIAGNOSTIC START ==="
Write-Log "System Model: $(Get-CimInstance Win32_ComputerSystem | Select-Object -ExpandProperty Model)"
Write-Log "BIOS Version: $(Get-CimInstance Win32_BIOS | Select-Object -ExpandProperty SMBIOSBIOSVersion)"

# Check WiFi Adapter Status
$wifi = Get-PnpDevice -Class Net | Where-Object { $_.FriendlyName -like "*9560*" }
if ($wifi) {
    Write-Log "WiFi Device: $($wifi.FriendlyName)"
    Write-Log "WiFi Status: $($wifi.Status)"
    Write-Log "WiFi InstanceId: $($wifi.InstanceId)"
    
    # Get detailed problem code
    $problem = Get-PnpDeviceProperty -InstanceId $wifi.InstanceId -KeyName DEVPKEY_Device_ProblemCode
    $problemStatus = Get-PnpDeviceProperty -InstanceId $wifi.InstanceId -KeyName DEVPKEY_Device_ProblemStatus
    Write-Log "Problem Code: $($problem.Data)"
    Write-Log "Problem Status: $($problemStatus.Data)"
} else {
    Write-Log "WARNING: Intel 9560 NOT FOUND in PnP devices"
}

# Check MEI Status
$mei = Get-PnpDevice -Class System | Where-Object { $_.FriendlyName -like "*Management Engine*" }
if ($mei) {
    Write-Log "MEI Device: $($mei.FriendlyName)"
    Write-Log "MEI Status: $($mei.Status)"
    Write-Log "MEI Driver Version: $(Get-PnpDeviceProperty -InstanceId $mei.InstanceId -KeyName DEVPKEY_Device_DriverVersion | Select-Object -ExpandProperty Data)"
} else {
    Write-Log "ERROR: Intel MEI NOT FOUND - This explains Code 10!"
}

# Check Critical Services
@("WlanSvc", "LMS") | ForEach-Object {
    $svc = Get-Service -Name $_ -ErrorAction SilentlyContinue
    if ($svc) {
        Write-Log "Service $_ : $($svc.Status) (StartType: $($svc.StartType))"
    } else {
        Write-Log "Service $_ : MISSING"
    }
}

Write-Log "=== DIAGNOSTIC COMPLETE: $logFile ==="