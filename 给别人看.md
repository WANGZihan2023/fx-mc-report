# 长期分享

## 仓库

https://github.com/WANGZihan2023/fx-mc-report（已是 Public）

## 若出现 “You do not have access to this app”

几乎一定是 **Streamlit 里把应用设成了私有**（会强制登录鉴权）。对方没账号就会失败；你用 qq 邮箱登录也可能对不上。

### 改成公开（必须由你点）

1. 打开 https://share.streamlit.io/ 并登录（用部署时的 **GitHub: wangzihan2023 / WANGZihan2023**）
2. 进入 **My apps**，点开 `fx-mc-report`（或你起的名字）
3. 右上角 **Share**（或 ⋮ → **Settings** → **Sharing**）
4. **Who can view this app** 选：
   - **This app is public and searchable**  
   或 **Anyone can view / Anyone with the link**
5. 保存，等约 30 秒
6. **无痕窗口**打开应用链接验证（不要登录也应能看）

正确公开后，链接类似：

`https://fx-mc-report.streamlit.app`  
（以你面板顶部显示的 URL 为准）

### 仍打不开时

- 先 **Sign out**，再用 GitHub `WANGZihan2023` 登录（不要只靠 qq 邮箱）
- 确认工作区（workspace）是部署该 app 的那个
- 或删掉 app 后按下面重新 Deploy，部署完立刻设为 Public

重新部署页：  
https://share.streamlit.io/deploy?repository=WANGZihan2023/fx-mc-report&branch=main&mainModule=app.py
