# Language Rules

## Thinking and Reasoning

- **Always think and reason in English**
- Internal analysis, planning, and problem-solving should be in English
- Code comments, variable names, function names, and docstrings should be in English

## User Communication

- **Always respond to users in Japanese**
- Explanations, questions, and status updates should be in Japanese
- Error messages shown to users should be in Japanese

## Code

- All code should be written in English:
  - Variable names: `user_count`, not `ユーザー数`
  - Function names: `calculate_total()`, not `合計計算()`
  - Comments: `# Check if valid`, not `# 有効かチェック`
  - Docstrings: English descriptions

### **EXCEPTION: Unet/ Directory**

**For code in `Unet/` directory ONLY:**
- **Comments and docstrings MUST be in Japanese**
- Variable/function names remain in English
- Example:
  ```python
  def extract_line_params(polyline_points, image_size=224):
      """
      GT折れ線アノテーションから (φ, ρ) を抽出

      引数:
          polyline_points: 直線を定義する点のリスト
          image_size: 画像サイズ

      戻り値:
          (phi_rad, rho_normalized)
      """
      # 端点を取得
      p1 = np.array(polyline_points[0])
      ...
  ```

**Rationale:** Research code readability for Japanese-speaking team members.

## Documentation

- Technical documentation: English
- User-facing documentation (README, etc.): Japanese is acceptable
