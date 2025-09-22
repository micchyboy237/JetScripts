Set-Content -Path "C:\Users\druiv\Desktop\Jet_Files\Jet_Windows_Workspace\cli_commands\Run-VramRamInfo.ps1" -Value @"
# Run-VramRamInfo.ps1
# Continuously display GPU and system RAM information in CSV format with newline separation

# Function to get system RAM info in CSV format
function Get-SystemRamInfo {
    `$os = Get-CimInstance -ClassName Win32_OperatingSystem
    `$totalRam = [math]::Round(`$os.TotalVisibleMemorySize / 1MB, 2)  # Convert KB to GB
    `$freeRam = [math]::Round(`$os.FreePhysicalMemory / 1MB, 2)      # Convert KB to GB
    `$usedRam = [math]::Round(`$totalRam - `$freeRam, 2)
    `$utilization = [math]::Round((`$usedRam / `$totalRam) * 100, 2)  # Calculate RAM utilization percentage
    return "0, System RAM, `$utilization %, `${usedRam} GB, `${totalRam} GB"
}

# Output CSV header once
Write-Output "index, name, utilization, used, total"

# Main loop to display GPU and RAM info
while (`$true) {
    # Get GPU information using nvidia-smi
    `$gpuInfo = nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader

    # Get RAM information in CSV format
    `$ramInfo = Get-SystemRamInfo

    # Display GPU and RAM info
    Write-Output `$gpuInfo
    Write-Output `$ramInfo
    Write-Output ""  # Add newline for separation between iterations

    # Wait for 1 second before refreshing
    Start-Sleep -Seconds 1
}
"@
