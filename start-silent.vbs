' qoder2api 后台静默启动脚本
' 通过 WScript.Shell 以后台隐藏窗口形式运行 run-background.cmd
Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "H:\qoder2api"
WshShell.Run "cmd /c run-background.cmd", 0, False
