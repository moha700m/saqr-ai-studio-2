; SAQR AI Studio - Windows one-click installer (Inno Setup 6.3+)
; Build:  python scripts/gen_inno_models.py && iscc /DAppVersion=0.1.0 installer\saqr.iss
;
; Flow: welcome -> AGPL licence -> detected hardware -> model licences -> folder -> tasks -> install
;       -> download models (SHA-256 verified) -> diagnostics (real backend verification) -> launch.
; Silent install (used by CI):  SAQR-AI-Setup.exe /VERYSILENT /SUPPRESSMSGBOXES /ACCEPTMODELLICENSES=1

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
#define AppName "SAQR AI Studio"
#define AppExe "SAQR AI Studio.exe"
#define AppPublisher "SAQR AI Studio contributors"
#define AppURL "https://github.com/moha700m/saqr-ai-studio-2"

[Setup]
AppId={{6E1F4C2A-5A1B-4E8E-9C61-5A9D2B7F3E10}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
LicenseFile=..\LICENSE
OutputDir=Output
OutputBaseFilename=SAQR-AI-Setup
SetupIconFile=..\desktop\saqr_desktop\assets\saqr.ico
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ExtraDiskSpaceRequired=1300000000
CloseApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "enhancer"; Description: "Also download the face enhancer (GFPGAN, 340 MB)"; GroupDescription: "AI models:"

[Files]
Source: "..\dist\SAQR AI Studio\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\LICENSE"; DestDir: "{app}"; DestName: "LICENSE.txt"; Flags: ignoreversion
Source: "..\MODEL_LICENSES.md"; DestDir: "{app}"; Flags: ignoreversion skipifsourcedoesntexist

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autoprograms}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Parameters: "--probe-json ""{localappdata}\{#AppName}\install_selftest.json"""; \
  StatusMsg: "Testing your hardware and the AI models..."; Flags: runhidden waituntilterminated
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"

[Code]
var
  ModelIds, ModelFiles, ModelUrls, ModelHashes: array[0..15] of String;
  ModelSizesMB: array[0..15] of Integer;
  ModelOptional: array[0..15] of Boolean;
  HwPage: TOutputMsgMemoWizardPage;
  ModelPage: TWizardPage;
  ModelMemo: TNewMemo;
  ModelAccept: TNewCheckBox;
  DownloadFailures: String;
  HwSummary: String;

#include "models.iss"

function WmiList(const Query, Prop: String): String;
var
  Locator, Service, Items, Item, Val: Variant;
  I: Integer;
  Str: String;
begin
  Result := '';
  try
    Locator := CreateOleObject('WbemScripting.SWbemLocator');
    Service := Locator.ConnectServer('.', 'root\CIMV2');
    Items := Service.ExecQuery(Query);
    for I := 0 to Items.Count - 1 do
    begin
      try
        Item := Items.ItemIndex(I);
        Val := Item.Properties_.Item(Prop).Value;
        Str := Val;
      except
        Str := '?';
      end;
      if Result <> '' then Result := Result + '; ';
      Result := Result + Trim(Str);
    end;
  except
    Result := '(could not query)';
  end;
end;

function ContainsText(const S, Sub: String): Boolean;
begin
  Result := Pos(Lowercase(Sub), Lowercase(S)) > 0;
end;

function DetectHardware: String;
var
  Cpu, Cores, Threads, Gpus, Ram, Backend: String;
  RamMB: Int64;
begin
  Cpu := WmiList('SELECT Name FROM Win32_Processor', 'Name');
  Cores := WmiList('SELECT NumberOfCores FROM Win32_Processor', 'NumberOfCores');
  Threads := WmiList('SELECT NumberOfLogicalProcessors FROM Win32_Processor', 'NumberOfLogicalProcessors');
  Gpus := WmiList('SELECT Name FROM Win32_VideoController', 'Name');
  Ram := WmiList('SELECT TotalPhysicalMemory FROM Win32_ComputerSystem', 'TotalPhysicalMemory');
  RamMB := StrToInt64Def(Ram, 0) div 1048576;
  if ContainsText(Gpus, 'NVIDIA') or ContainsText(Gpus, 'GeForce') or ContainsText(Gpus, 'RTX') or
     ContainsText(Gpus, 'Radeon') or ContainsText(Gpus, 'AMD') or ContainsText(Gpus, 'Intel') or
     ContainsText(Gpus, 'Arc') then
    Backend := 'GPU acceleration through DirectML (DirectX 12). It is tested after installation; if your GPU ' +
               'or driver cannot run it, SAQR switches to the CPU automatically.'
  else
    Backend := 'Optimised CPU (no supported GPU found). Live camera processing will be slow without a GPU.';
  HwSummary := 'Processor:  ' + Cpu + #13#10 +
               'Cores:      ' + Cores + ' cores / ' + Threads + ' threads' + #13#10 +
               'Graphics:   ' + Gpus + #13#10 +
               'Memory:     ' + IntToStr(Integer(RamMB div 1024)) + '.' + IntToStr(Integer((RamMB mod 1024) * 10 div 1024)) + ' GB' + #13#10 +
               'Windows:    ' + GetWindowsVersionString + ' (64-bit)' + #13#10#13#10 +
               'Selected AI engine:' + #13#10 + Backend;
  if RamMB < 7600 then
    HwSummary := HwSummary + #13#10#13#10 + 'Note: less than 8 GB RAM was detected. SAQR will work, ' +
                 'but use the Fast preset.';
  Result := HwSummary;
end;

procedure ModelAcceptClick(Sender: TObject);
begin
  WizardForm.NextButton.Enabled := True;
end;

procedure InitializeWizard;
var
  I, Total: Integer;
  S: String;
begin
  FillModels;
  HwPage := CreateOutputMsgMemoPage(wpLicense, 'Your computer',
    'SAQR checked your hardware to choose the best AI engine.', 'Detected hardware:', DetectHardware);

  ModelPage := CreateCustomPage(HwPage.ID, 'AI models',
    'SAQR downloads its AI models from their public source and verifies every file (SHA-256).');
  ModelMemo := TNewMemo.Create(ModelPage);
  ModelMemo.Parent := ModelPage.Surface;
  ModelMemo.Left := 0; ModelMemo.Top := 0;
  ModelMemo.Width := ModelPage.SurfaceWidth;
  ModelMemo.Height := ModelPage.SurfaceHeight - ScaleY(48);
  ModelMemo.ScrollBars := ssVertical;
  ModelMemo.ReadOnly := True;
  Total := 0;
  S := 'The face analysis and face swap models (InsightFace: SCRFD, ArcFace, 2d106, inswapper) are licensed for ' +
       'NON-COMMERCIAL RESEARCH USE ONLY. Commercial use requires a licence from InsightFace.' + #13#10 +
       'The NSFW safety filter (Apache-2.0) and the GFPGAN enhancer (Apache-2.0) are permissively licensed.' + #13#10 +
       'Use SAQR only with your own face or faces you have explicit permission to use.' + #13#10#13#10 + 'Files:' + #13#10;
  for I := 0 to ModelCount - 1 do
  begin
    S := S + '  ' + ModelFiles[I] + '  (' + IntToStr(ModelSizesMB[I]) + ' MB)';
    if ModelOptional[I] then S := S + '  [optional]';
    S := S + #13#10 + '    ' + ModelUrls[I] + #13#10;
    if not ModelOptional[I] then Total := Total + ModelSizesMB[I];
  end;
  S := S + #13#10 + 'Required download: about ' + IntToStr(Total) + ' MB.';
  ModelMemo.Text := S;
  ModelAccept := TNewCheckBox.Create(ModelPage);
  ModelAccept.Parent := ModelPage.Surface;
  ModelAccept.Top := ModelMemo.Top + ModelMemo.Height + ScaleY(12);
  ModelAccept.Width := ModelPage.SurfaceWidth;
  ModelAccept.Height := ScaleY(20);
  ModelAccept.Caption := 'I accept these model licences - download the models during installation';
  ModelAccept.Checked := ExpandConstant('{param:ACCEPTMODELLICENSES|0}') = '1';
  ModelAccept.OnClick := @ModelAcceptClick;
end;

function DataDir: String;
begin
  Result := ExpandConstant('{localappdata}\{#AppName}');
end;

function OnDownloadProgress(const Url, FileName: String; const Progress, ProgressMax: Int64): Boolean;
begin
  if ProgressMax > 0 then
  begin
    WizardForm.ProgressGauge.Max := 1000;
    WizardForm.ProgressGauge.Position := Integer(Progress * 1000 div ProgressMax);
  end;
  WizardForm.StatusLabel.Caption := 'Downloading ' + FileName + ' - ' +
    IntToStr(Integer(Progress div 1048576)) + ' / ' + IntToStr(Integer(ProgressMax div 1048576)) + ' MB';
  Result := True;
end;

procedure DownloadModels;
var
  I: Integer;
  Dest, Accepted: String;
begin
  ForceDirectories(DataDir + '\models');
  Accepted := '';
  for I := 0 to ModelCount - 1 do
  begin
    if ModelOptional[I] and not WizardIsTaskSelected('enhancer') then Continue;
    Accepted := Accepted + ModelIds[I] + #13#10;
    Dest := DataDir + '\models\' + ModelFiles[I];
    if FileExists(Dest) and (GetSHA256OfFile(Dest) = ModelHashes[I]) then Continue;
    try
      DownloadTemporaryFile(ModelUrls[I], ModelFiles[I], ModelHashes[I], @OnDownloadProgress);
      if not CopyFile(ExpandConstant('{tmp}\') + ModelFiles[I], Dest, False) then
        DownloadFailures := DownloadFailures + ModelFiles[I] + ' (could not save)' + #13#10;
      DeleteFile(ExpandConstant('{tmp}\') + ModelFiles[I]);
    except
      DownloadFailures := DownloadFailures + ModelFiles[I] + ': ' + GetExceptionMessage + #13#10;
    end;
  end;
  // records the licence acceptance made on the wizard page, read by the app's Model Manager
  SaveStringToFile(DataDir + '\accepted_licenses.txt', Accepted, False);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssPostInstall) and ModelAccept.Checked then
  begin
    WizardForm.StatusLabel.Caption := 'Downloading AI models...';
    DownloadModels;
    if DownloadFailures <> '' then
      SuppressibleMsgBox('Some models could not be downloaded:' + #13#10 + DownloadFailures + #13#10 +
        'You can download them later from the Models page in SAQR AI Studio.', mbInformation, MB_OK, IDOK);
  end;
end;

procedure CurPageChanged(CurPageID: Integer);
var
  Summary: AnsiString;
begin
  if CurPageID = ModelPage.ID then
    WizardForm.NextButton.Enabled := True;
  if CurPageID = wpFinished then
  begin
    if LoadStringFromFile(DataDir + '\install_selftest.txt', Summary) then
      WizardForm.FinishedLabel.Caption := 'SAQR AI Studio is installed and was tested on your computer:' + #13#10#13#10 + String(Summary) + #13#10#13#10 + 'Click Finish to start.'
    else if not ModelAccept.Checked then
      WizardForm.FinishedLabel.Caption := 'SAQR AI Studio is installed. Open the Models page to download ' +
        'the AI models before first use.';
  end;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if (CurPageID = ModelPage.ID) and not ModelAccept.Checked then
    Result := SuppressibleMsgBox('Continue without downloading the AI models? You can download them later from ' +
      'the Models page.', mbConfirmation, MB_YESNO, IDYES) = IDYES;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    if DirExists(ExpandConstant('{localappdata}\{#AppName}')) then
      if SuppressibleMsgBox('Also delete the downloaded AI models and your SAQR settings?',
           mbConfirmation, MB_YESNO, IDNO) = IDYES then
        DelTree(ExpandConstant('{localappdata}\{#AppName}'), True, True, True);
end;
