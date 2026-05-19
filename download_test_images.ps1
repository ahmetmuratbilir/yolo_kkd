$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$OutDir = Join-Path $ProjectDir "test_images"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$Queries = @(
    "construction worker safety vest hard hat",
    "industrial worker reflective vest helmet gloves",
    "worker safety glasses gloves vest",
    "construction worker PPE goggles gloves",
    "high visibility vest worker helmet",
    "road worker reflective vest hard hat",
    "factory worker safety vest helmet",
    "engineer safety vest hardhat"
)

$MaxImages = 30
$Downloaded = 0
$Seen = @{}

$PexelsPages = @(
    "https://www.pexels.com/photo/construction-worker-with-hard-hat-and-safety-vest-35390341/",
    "https://www.pexels.com/photo/man-in-yellow-vest-wearing-hard-hat-and-gloves-9162010/",
    "https://www.pexels.com/photo/photograph-of-men-with-hard-hats-wearing-reflective-vests-11790051/",
    "https://www.pexels.com/photo/men-working-on-a-construction-site-while-wearing-hard-hats-10202856/",
    "https://www.pexels.com/photo/workers-wearing-safety-vest-and-hard-hat-walking-on-the-dry-ground-8961154/",
    "https://www.pexels.com/photo/construction-workers-in-safety-gear-on-site-34670929/",
    "https://www.pexels.com/photo/construction-worker-holding-hard-hat-outdoors-34670931/",
    "https://www.pexels.com/photo/a-man-in-a-yellow-vest-holding-a-hard-hat-28196526/",
    "https://www.pexels.com/photo/backview-of-man-in-reflective-vest-and-hard-hat-11959727/",
    "https://www.pexels.com/photo/construction-workers-in-blue-vests-and-hard-hats-36574301/",
    "https://www.pexels.com/photo/woman-in-work-clothes-wearing-hard-hat-and-gloves-8961396/",
    "https://www.pexels.com/photo/a-man-wearing-construction-safety-gear-4981771/",
    "https://www.pexels.com/photo/close-up-photo-of-person-holding-hardhat-8487720/"
)

$PexelsPhotoIds = @(
    35390341, 9162010, 11790051, 10202856, 8961154,
    34670929, 34670931, 28196526, 11959727, 36574301,
    8961396, 4981771, 8487720, 8487797, 3979016,
    34670925, 32467381, 8488021, 8293645, 15834397,
    9258892, 8488020, 35390338, 16440411, 8487396,
    8486907, 5422490, 8487761, 14147712
)

foreach ($PhotoId in $PexelsPhotoIds) {
    if ($Downloaded -ge $MaxImages) { break }
    $ImageUrl = "https://images.pexels.com/photos/$PhotoId/pexels-photo-$PhotoId.jpeg?auto=compress&cs=tinysrgb&w=1280"
    if ($Seen.ContainsKey($ImageUrl)) { continue }
    $Seen[$ImageUrl] = $true

    $OutPath = Join-Path $OutDir ("{0:D2}_pexels_$PhotoId.jpg" -f ($Downloaded + 1))
    try {
        Invoke-WebRequest -Uri $ImageUrl -OutFile $OutPath -TimeoutSec 30 -Headers @{ "User-Agent" = "Mozilla/5.0" }
        if ((Test-Path $OutPath) -and ((Get-Item $OutPath).Length -gt 10000)) {
            $Downloaded += 1
            Write-Host "[download_test_images] $Downloaded/$MaxImages -> $OutPath"
        } else {
            Remove-Item -LiteralPath $OutPath -ErrorAction SilentlyContinue
        }
    } catch {
        Remove-Item -LiteralPath $OutPath -ErrorAction SilentlyContinue
    }
}

foreach ($PageUrl in $PexelsPages) {
    if ($Downloaded -ge $MaxImages) { break }
    try {
        $Page = Invoke-WebRequest -Uri $PageUrl -TimeoutSec 30 -Headers @{ "User-Agent" = "Mozilla/5.0" }
        $Match = [regex]::Match($Page.Content, '<meta\s+property="og:image"\s+content="([^"]+)"')
        if (-not $Match.Success) {
            $Match = [regex]::Match($Page.Content, '<meta\s+content="([^"]+)"\s+property="og:image"')
        }
        if (-not $Match.Success) { continue }
        $ImageUrl = $Match.Groups[1].Value -replace '&amp;', '&'
        if ($Seen.ContainsKey($ImageUrl)) { continue }
        $Seen[$ImageUrl] = $true

        $OutPath = Join-Path $OutDir ("{0:D2}_pexels_ppe.jpg" -f ($Downloaded + 1))
        Invoke-WebRequest -Uri $ImageUrl -OutFile $OutPath -TimeoutSec 30 -Headers @{ "User-Agent" = "Mozilla/5.0" }
        if ((Test-Path $OutPath) -and ((Get-Item $OutPath).Length -gt 10000)) {
            $Downloaded += 1
            Write-Host "[download_test_images] $Downloaded/$MaxImages -> $OutPath"
        } else {
            Remove-Item -LiteralPath $OutPath -ErrorAction SilentlyContinue
        }
    } catch {
        continue
    }
}

foreach ($Query in $Queries) {
    if ($Downloaded -ge $MaxImages) { break }

    $Encoded = [System.Uri]::EscapeDataString($Query)
    $ApiUrl = "https://api.openverse.engineering/v1/images/?q=$Encoded&page_size=50"
    Write-Host "[download_test_images] Arama: $Query"

    try {
        $Response = Invoke-RestMethod -Uri $ApiUrl -TimeoutSec 30
    } catch {
        Write-Host "[download_test_images] Arama hatasi: $($_.Exception.Message)" -ForegroundColor Yellow
        continue
    }

    foreach ($Item in $Response.results) {
        if ($Downloaded -ge $MaxImages) { break }
        $Url = $Item.url
        if (-not $Url) { $Url = $Item.thumbnail }
        if (-not $Url -or $Seen.ContainsKey($Url)) { continue }
        $Seen[$Url] = $true

        $SafeName = ($Item.title -replace '[^a-zA-Z0-9_-]+', '_')
        if (-not $SafeName) { $SafeName = "ppe_image" }
        if ($SafeName.Length -gt 48) { $SafeName = $SafeName.Substring(0, 48) }

        $OutPath = Join-Path $OutDir ("{0:D2}_{1}.jpg" -f ($Downloaded + 1), $SafeName)
        $CandidateUrls = @($Url)
        if ($Item.thumbnail -and $Item.thumbnail -ne $Url) {
            $CandidateUrls += $Item.thumbnail
        }

        foreach ($CandidateUrl in $CandidateUrls) {
            try {
                Invoke-WebRequest -Uri $CandidateUrl -OutFile $OutPath -TimeoutSec 30 -Headers @{ "User-Agent" = "Mozilla/5.0" }
            } catch {
                Remove-Item -LiteralPath $OutPath -ErrorAction SilentlyContinue
                continue
            }

            if ((Test-Path $OutPath) -and ((Get-Item $OutPath).Length -gt 10000)) {
                $Downloaded += 1
                Write-Host "[download_test_images] $Downloaded/$MaxImages -> $OutPath"
                break
            } else {
                Remove-Item -LiteralPath $OutPath -ErrorAction SilentlyContinue
            }
        }
    }
}

Write-Host "[download_test_images] Indirilen gorsel sayisi: $Downloaded"
