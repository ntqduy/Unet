# Thuật Toán Pruning Tổng Quát

Tài liệu này tóm tắt ba hướng pruning trong code: **Blueprint**, **Middle Conv2**, và **Full-Block**. Các phương pháp chọn ngưỡng như Static, Kneedle, Otsu, GMM chỉ được mô tả ở mức ý tưởng, không triển khai chi tiết.

## 1. Chọn Ngưỡng Pruning

Input:

- `S = {s1, s2, ..., sC}`: importance score của `C` channel.
- `M`: phương pháp chọn ngưỡng, gồm `static`, `kneedle`, `otsu`, hoặc `gmm`.

Thuật toán:

1. Tính ngưỡng `tau` hoặc tập channel cần giữ:
   - **Static**: giữ top `ceil((1 - r) * C)` channel có importance cao nhất.
   - **Kneedle**: sắp xếp importance tăng dần, chuẩn hóa về `[0, 1]`, chọn điểm gối làm ngưỡng.
   - **Otsu**: chọn ngưỡng tối đa hóa phương sai giữa hai nhóm importance thấp/cao.
   - **GMM**: fit hai Gaussian lên importance score, chọn giao điểm hai phân phối làm ngưỡng.
2. Với Kneedle/Otsu/GMM, giữ các channel thỏa:

```text
score >= tau
```

3. Nếu số channel giữ lại nhỏ hơn giới hạn tối thiểu, ép giữ thêm các channel có score cao nhất.
4. Output:
   - `keep_indices`
   - `pruned_indices`
   - `num_keep`
   - `pruning_threshold`

## 2. Blueprint Pruning

Mục tiêu: tạo cấu hình số channel cho các stage chính của encoder/student.

Input:

- Teacher model.
- Danh sách stage mục tiêu, ví dụ:
  - `stem, down1, down2, down3, down4`
  - hoặc `conv1, layer1, layer2, layer3, layer4`
- Phương pháp pruning `M`.

Thuật toán:

```text
for mỗi stage trong target_modules:
    tìm convolution chính đại diện output channel của stage

    nếu có normalization layer tương ứng:
        importance = abs(norm.weight)
    ngược lại:
        importance = L1_norm(conv.weight theo từng output channel)

    áp dụng phương pháp M để chọn keep_indices
    keep_channels = số channel được giữ lại
    lưu keep_channels vào channel_config

output channel_config = (c1, c2, c3, c4, c5)
```

Ý nghĩa:

- Prune ở mức output channel của từng stage.
- Kết quả là một blueprint 5 stage để xây student nhỏ hơn.
- Đây là cách tổng quát nhất, ít phụ thuộc vào cấu trúc bottleneck chi tiết.

## 3. Middle Conv2 Pruning

Mục tiêu: prune phần giữa của ResNet bottleneck, nhưng giữ nguyên output block để ít phá residual connection.

Áp dụng cho các method:

- `middle_static`
- `middle_kneedle`
- `middle_otsu`
- `middle_gmm`

Với mỗi bottleneck block:

```text
conv1 -> conv2 -> conv3
```

Code chỉ prune:

```text
conv2 output == conv3 input
```

Thuật toán:

```text
for mỗi bottleneck block trong layer1..layer4:
    lấy importance của conv2_out
    lấy importance của conv3_in

    chuẩn hóa từng importance về cùng thang đo
    importance = mean(normalize(conv2_out), normalize(conv3_in))

    áp dụng phương pháp M để chọn keep_indices

    prune channel ở conv2 output và conv3 input
    giữ nguyên boundary conv1 và conv3 output

    lưu:
        kept_middle_channels
        kept_channel_indices
        pruned_channel_indices
```

Ý nghĩa:

- Chỉ giảm internal width của bottleneck.
- `conv1` và `conv3` được bảo vệ để giữ ổn định shape đầu vào/đầu ra block.
- Phù hợp khi muốn giảm tham số nhưng hạn chế thay đổi residual topology.

## 4. Full-Block Pruning

Mục tiêu: prune toàn bộ bottleneck block, gồm cả internal channel và output channel.

Áp dụng cho các method:

- `full_static`
- `full_kneedle`
- `full_otsu`
- `full_gmm`

Với mỗi bottleneck block:

```text
conv1 -> conv2 -> conv3 -> next block
```

Thuật toán:

```text
for mỗi bottleneck block trong layer1..layer4:
    tính importance cho cạnh conv1 -> conv2:
        I1 = mean(normalize(conv1_out), normalize(conv2_in))

    tính importance cho cạnh conv2 -> conv3:
        I2 = mean(normalize(conv2_out), normalize(conv3_in))

    tính importance cho output block:
        I3 = mean(normalize(conv3_out), normalize(next_input))

    áp dụng phương pháp M riêng cho I1, I2, I3

    lấy:
        conv1_keep_indices từ I1
        conv2_keep_indices từ I2
        output_keep_indices từ I3

    nếu block dùng identity residual:
        output channel phải tương thích với input channel hiện tại

    nếu input/output channel khác nhau:
        cần residual projection hoặc rebuild shape

    lưu kế hoạch pruning của block:
        kept_internal_channels
        kept_conv2_channels
        kept_output_channels
        kept/pruned indices
```

Ý nghĩa:

- Đây là chiến lược pruning mạnh nhất.
- Không chỉ giảm width bên trong block mà còn giảm output channel của block.
- Vì output bị prune, residual connection và decoder shape có thể phải được rebuild.

## 5. So Sánh Nhanh

| Chiến lược | Vị trí prune | Mức thay đổi kiến trúc | Độ mạnh |
|---|---|---:|---:|
| Blueprint | Output channel của stage | Vừa | Trung bình |
| Middle Conv2 | Chỉ `conv2_out / conv3_in` trong bottleneck | Nhỏ | Vừa |
| Full-Block | `conv1`, `conv2`, `conv3 output`, residual path | Lớn | Mạnh |

## 6. Ghi Chú Theo Code

- Importance ưu tiên lấy từ normalization weight nếu có, nếu không dùng L1 norm của convolution filter.
- Dynamic methods như Kneedle/Otsu/GMM dùng threshold tự động.
- Static methods dùng prune ratio cố định.
- Với dynamic methods, code có ràng buộc giữ tối thiểu `minimum_channels` hoặc `dynamic_min_keep_ratio`.
- `figure3_thresholding_methods` và các figure liên quan dùng các `importance` và `pruning_threshold` này để vẽ histogram và đường ngưỡng.
