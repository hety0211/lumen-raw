; Build with Inno Setup 6.7+ / 7 from the source directory:
;   ISCC.exe /DAppBuild=dist\LumenRAW installer.iss
; AppVersion must equal lumen/__init__.py __version__ (checked by tests/test_v13_release.py).
; 2.1.1: /DUpdatePackage builds LumenRAW-<version>-Update.exe, the program without the AI models
; (dist\LumenRAW\models), which the editor downloads and runs to update itself.
#ifndef AppBuild
  #define AppBuild "dist\LumenRAW"
#endif
#ifndef AppVersion
  #define AppVersion "2.1.2"
#endif
#ifdef UpdatePackage
  #define PackageKind "Update"
#else
  #define PackageKind "Setup"
#endif
#define PackageDir "v" + StringChange(AppVersion, ".", "")

[Setup]
AppId={{F1CA8FF7-EB54-4B53-81E3-4CC183C8C1B9}
AppName=LUMEN RAW
AppVersion={#AppVersion}
VersionInfoVersion={#AppVersion}
AppPublisher=LUMEN RAW
DefaultDirName={localappdata}\Programs\LUMEN RAW
DefaultGroupName=LUMEN RAW
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.19045
OutputDir=.publish\{#PackageDir}\packages
OutputBaseFilename=LumenRAW-{#AppVersion}-{#PackageKind}
Compression=lzma2/normal
SolidCompression=yes
WizardStyle=modern
SetupIconFile=assets\lumen.ico
UninstallDisplayIcon={app}\LumenRAW.exe
LicenseFile=LICENSE
CloseApplications=yes
RestartApplications=no
Uninstallable=not PortableMode
CreateUninstallRegKey=not PortableMode
UsePreviousAppDir=not PortableMode

[Languages]
; 1.5.1: the same languages as the editor's 语言 / Language menu.  The choice is written to
; settings.ini, so the editor starts in it.  Names: zhcn / en as in earlier installers (their
; previous choice stays preselected on upgrade); the editor maps zhcn / zhtw to zh_CN / zh_TW.
Name: "zhcn"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"
Name: "zhtw"; MessagesFile: "compiler:Languages\ChineseTraditional.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"
Name: "ja"; MessagesFile: "compiler:Languages\Japanese.isl"
Name: "ko"; MessagesFile: "compiler:Languages\Korean.isl"
Name: "de"; MessagesFile: "compiler:Languages\German.isl"
Name: "fr"; MessagesFile: "compiler:Languages\French.isl"
Name: "es"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "ru"; MessagesFile: "compiler:Languages\Russian.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; Check: not PortableMode

[Dirs]
Name: "{localappdata}\LUMEN RAW"; Flags: uninsneveruninstall; Check: not PortableMode

[INI]
; Read by lumen/i18n.py; the editor's language menu writes the same key.
Filename: "{localappdata}\LUMEN RAW\settings.ini"; Section: "General"; Key: "language"; String: "{language}"; Check: not PortableMode

[Files]
#ifdef UpdatePackage
; The models folder of the installation stays as it is; the editor fetches models it lacks.
Source: "{#AppBuild}\*"; DestDir: "{app}"; Excludes: "\models"; Flags: ignoreversion recursesubdirs createallsubdirs
#else
Source: "{#AppBuild}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
#endif

[InstallDelete]
; Upgrades install over the previous version (same AppId); drop its runtime first so
; no stale DLL or model from an older build can shadow the new one.  2.1.1: the AI models live in
; {app}\models (lumen/ai_models.py), which is not part of the runtime and is kept.
Type: filesandordirs; Name: "{app}\_internal"; Check: not PortableMode
Type: files; Name: "{app}\LumenARW.exe"; Check: not PortableMode
Type: files; Name: "{group}\Lumen ARW.lnk"; Check: not PortableMode
Type: files; Name: "{userdesktop}\Lumen ARW.lnk"; Check: not PortableMode

[Icons]
Name: "{group}\LUMEN RAW"; Filename: "{app}\LumenRAW.exe"; Check: not PortableMode
Name: "{userdesktop}\LUMEN RAW"; Filename: "{app}\LumenRAW.exe"; Tasks: desktopicon; Check: not PortableMode

[UninstallDelete]
; 1.8.0: the speech model downloaded on first use (lumen/speech.py) is a re-downloadable cache.
Type: filesandordirs; Name: "{localappdata}\LUMEN RAW\models"; Check: not PortableMode
; 2.0.0: the preview cache (lumen/preview_cache.py) is rebuilt when needed; the library catalog
; (library.db) and the ratings (catalog.jsonl) are the user's and stay.
Type: filesandordirs; Name: "{localappdata}\LUMEN RAW\cache\previews"; Check: not PortableMode
; 2.1.1: models the editor downloaded itself are not in the uninstall log.
Type: filesandordirs; Name: "{app}\models"; Check: not PortableMode

[Run]
Filename: "{app}\LumenRAW.exe"; Description: "{cm:LaunchProgram,LUMEN RAW}"; Flags: nowait postinstall skipifsilent
; 2.1.1: an update started from the editor (/SILENT /RELAUNCH=1) opens it again when it is done.
Filename: "{app}\LumenRAW.exe"; Flags: nowait; Check: RelaunchRequested

[Code]
const
  SYNCHRONIZE = $00100000;

function OpenProcess(DesiredAccess: DWORD; InheritHandle: BOOL; ProcessId: DWORD): THandle;
  external 'OpenProcess@kernel32.dll stdcall';
function WaitForSingleObject(Handle: THandle; Milliseconds: DWORD): DWORD;
  external 'WaitForSingleObject@kernel32.dll stdcall';
function CloseHandle(Handle: THandle): BOOL;
  external 'CloseHandle@kernel32.dll stdcall';

function PortableMode: Boolean;
begin
  Result := ExpandConstant('{param:PORTABLE|0}') = '1';
end;

function RelaunchRequested: Boolean;
begin
  Result := (ExpandConstant('{param:RELAUNCH|0}') = '1') and not PortableMode;
end;

function InitializeSetup(): Boolean;
var
  Pid: Integer;
  Process: THandle;
begin
  Result := True;
  { 2.1.1: the editor starts its update with /WAITPID=<its process id> and then quits; wait (at
    most a minute) until it has exited, so that none of its files are still in use. }
  Pid := StrToIntDef(ExpandConstant('{param:WAITPID|0}'), 0);
  if Pid > 0 then
  begin
    Process := OpenProcess(SYNCHRONIZE, False, Pid);
    if Process <> 0 then
    begin
      WaitForSingleObject(Process, 60000);
      CloseHandle(Process);
    end;
  end;
end;

procedure KeepFile(const Name, OldFolder, NewFolder: String);
begin
  if FileExists(OldFolder + '\' + Name) and not FileExists(NewFolder + '\' + Name) then
    if not RenameFile(OldFolder + '\' + Name, NewFolder + '\' + Name) then
      Log('Could not keep ' + Name + ' of the previous version');
end;

#ifdef UpdatePackage
procedure KeepModels(const OldFolder, NewFolder: String);
var
  Found: TFindRec;
begin
  if FindFirst(OldFolder + '\*.onnx', Found) then
  begin
    try
      if ForceDirectories(NewFolder) then
        repeat
          if not RenameFile(OldFolder + '\' + Found.Name, NewFolder + '\' + Found.Name) then
            Log('Could not keep the model ' + Found.Name + ' of the previous version');
        until not FindNext(Found);
    finally
      FindClose(Found);
    end;
  end;
end;
#endif

procedure CurStepChanged(CurStep: TSetupStep);
var
  OldFolder, NewFolder: String;
begin
  { 1.8.0: the speech model is downloaded on first use instead of installed.  Move the copy an
    earlier version installed to the download folder before [InstallDelete] removes _internal. }
  if (CurStep = ssInstall) and not PortableMode then
  begin
    OldFolder := ExpandConstant('{app}\_internal\assets\models\sensevoice');
    NewFolder := ExpandConstant('{localappdata}\LUMEN RAW\models\sensevoice');
    if FileExists(OldFolder + '\model.int8.onnx') and ForceDirectories(NewFolder) then
    begin
      KeepFile('model.int8.onnx', OldFolder, NewFolder);
      KeepFile('tokens.txt', OldFolder, NewFolder);
    end;
#ifdef UpdatePackage
    { 2.1.1: the update package carries no models.  Up to 2.1.0 they were part of _internal; hand
      them to the editor (models\import), which keeps the ones it still needs (lumen/ai_models.py). }
    KeepModels(ExpandConstant('{app}\_internal\assets\models'), ExpandConstant('{app}\models\import'));
#endif
  end;
end;
