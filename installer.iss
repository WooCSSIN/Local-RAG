; ==================================================
; Inno Setup script — Local RAG Chatbot v2
; ==================================================
; Cài Inno Setup: https://jrsoftware.org/isinfo.php
; Compile script này để tạo Setup.exe
; ==================================================

#define MyAppName "Local RAG Chatbot"
#define MyAppVersion "2.0"
#define MyAppPublisher "WooCSSIN"
#define MyAppURL "https://github.com/WooCSSIN/Local-RAG"
#define MyAppExeName "Local-RAG.exe"
#define SourceDir "D:\Local-RAG-Build\dist\Local-RAG"

[Setup]
AppId={{LocalRAGChatbot-v2}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DisableProgramGroupPage=yes
LicenseFile=LICENSE
OutputDir=D:\Local-RAG-Build\installer
OutputBaseFilename=Local-RAG-Chatbot-Setup
SetupIconFile=
Compression=lzma
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\qdrant_data"
Type: filesandordirs; Name: "{app}\chat_sessions"
