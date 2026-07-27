echo ==================== SECTION 1: DEVICE STATUS CHECK ====================
pnputil /enum-devices /class net /problem
echo.
echo ==================== SECTION 2: DRIVER DETAILS ====================
pnputil /enum-devices /class net /drivers
echo.
echo ==================== SECTION 3: NETWORK ADAPTER DRIVER QUERY ====================
driverquery /v | findstr /i "wireless\|wlan\|wi-fi"
echo.
echo ==================== SECTION 4: PROBLEMATIC DEVICE HARDWARE ID ====================
wmic path Win32_PnPEntity where "Status='Error'" get DeviceID,Name,HardwareID /format:list
echo.
echo ==================== SECTION 5: POWERSHELL DEVICE DETAILS ====================
powershell "Get-PnpDevice -Status Error | Where-Object {$_.Class -eq 'Net'} | Format-List Name,InstanceId,Status,Problem,Class"
echo.