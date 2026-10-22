; Build with Inno Setup 6.7+ / 7 from the source directory:
;   ISCC.exe /DAppBuild=dist\LumenRAW installer.iss
; AppVersion must equal lumen/__init__.py __version__ (checked by tests/test_v13_release.py).
#ifndef AppBuild
  #define AppBuild "dist\LumenRAW"
#endif
#ifndef AppVersion
  #define AppVersion "2.1.0"
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
OutputBaseFilename=LumenRAW-{#AppVersion}-Setup
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
Source: "{#AppBuild}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
; Upgrades install over the previous version (same AppId); drop its runtime first so
; no stale DLL or model from an older build can shadow the new one.
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

[Run]
Filename: "{app}\LumenRAW.exe"; Description: "{cm:LaunchProgram,LUMEN RAW}"; Flags: nowait postinstall skipifsilent

[Code]
function PortableMode: Boolean;
begin
  Result := ExpandConstant('{param:PORTABLE|0}') = '1';
end;

procedure KeepFile(const Name, OldFolder, NewFolder: String);
begin
  if FileExists(OldFolder + '\' + Name) and not FileExists(NewFolder + '\' + Name) then
    if not RenameFile(OldFolder + '\' + Name, NewFolder + '\' + Name) then
      Log('Could not keep ' + Name + ' of the previous version');
end;

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
  end;
end;
