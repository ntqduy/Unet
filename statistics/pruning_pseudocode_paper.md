# Pseudocode Cho Các Thuật Toán Pruning

## Notation

- `T`: teacher model.
- `M`: pruning method, ví dụ `Static`, `Kneedle`, `Otsu`, `GMM`.
- `S`: vector importance score của các channel.
- `tau`: pruning threshold.
- `K`: tập channel được giữ lại.
- `P`: tập channel bị prune.
- `Normalize(.)`: chuẩn hóa min-max hoặc chia theo giá trị lớn nhất.
- `SelectThreshold(S, M)`: hàm chọn ngưỡng theo phương pháp `M`.

## Algorithm 1: Threshold-Based Channel Selection

```text
Input:
    S = {s1, s2, ..., sC}: importance scores
    M: thresholding method
    r: static pruning ratio, only used when M = Static
    min_keep: minimum number of channels to keep

Output:
    K: kept channel indices
    P: pruned channel indices
    tau: pruning threshold

Procedure:
    if M = Static then
        k <- ceil((1 - r) * C)
        K <- indices of top-k largest scores in S
        tau <- minimum score among channels in K
    else
        tau <- SelectThreshold(S, M)
        K <- {i | S[i] >= tau}

        if |K| < min_keep then
            K <- indices of top min_keep largest scores in S
        end if
    end if

    P <- {1, 2, ..., C} \ K

    return K, P, tau
```

`SelectThreshold(S, M)` có thể là:

```text
Kneedle:
    sort S in ascending order
    normalize x-axis and sorted scores to [0, 1]
    choose tau at the point with maximum curve deviation

Otsu:
    build histogram of S
    choose tau that maximizes between-class variance

GMM:
    fit a two-component Gaussian mixture on S
    choose tau at the intersection of the two Gaussian components
```

## Algorithm 2: Blueprint Pruning

```text
Input:
    T: teacher model
    target_modules: main encoder stages
    M: pruning method

Output:
    channel_config: number of kept channels for each stage
    blueprint: pruning metadata for student construction

Procedure:
    channel_config <- empty list
    blueprint <- empty list

    for each module m in target_modules do
        L <- primary convolution layer of m

        if L does not exist then
            continue
        end if

        if a normalization layer corresponding to L exists then
            S <- absolute value of normalization weights
        else
            S <- L1 norm of convolution filters in L
        end if

        K, P, tau <- Threshold-Based Channel Selection(S, M)

        c_keep <- |K|
        append c_keep to channel_config

        save into blueprint:
            module name
            importance scores S
            kept indices K
            pruned indices P
            threshold tau
            number of kept channels c_keep
    end for

    return channel_config, blueprint
```

Paper-level description:

```text
Blueprint pruning estimates one representative channel configuration for each
major encoder stage. Each stage is compressed according to the selected
importance threshold, and the resulting channel_config is used to build the
student architecture.
```

## Algorithm 3: Middle Conv2 Pruning

```text
Input:
    T: teacher ResNet-based model
    M: pruning method

Output:
    middle_prune_plan: pruning plan for bottleneck middle channels

Procedure:
    middle_prune_plan <- empty list

    for each ResNet stage in {layer1, layer2, layer3, layer4} do
        for each bottleneck block b in stage do
            identify:
                conv1_b, conv2_b, conv3_b

            S_out <- importance of conv2_b output channels
            S_in  <- importance of conv3_b input channels

            S <- mean(Normalize(S_out), Normalize(S_in))

            K, P, tau <- Threshold-Based Channel Selection(S, M)

            prune channels in:
                conv2_b output
                conv3_b input

            keep unchanged:
                conv1_b boundary
                conv3_b output boundary

            save into middle_prune_plan:
                block name
                kept middle channel indices K
                pruned middle channel indices P
                threshold tau
                kept middle channel count |K|
        end for
    end for

    return middle_prune_plan
```

Paper-level description:

```text
Middle Conv2 pruning compresses only the internal width of each ResNet
bottleneck. The output boundary of the block is preserved, which keeps residual
connections and decoder interfaces stable.
```

## Algorithm 4: Full-Block Pruning

```text
Input:
    T: teacher ResNet-based model
    M: pruning method

Output:
    full_prune_plan: pruning plan for complete bottleneck blocks

Procedure:
    full_prune_plan <- empty list
    previous_output_indices <- all input channels of the first block

    for each ResNet stage in {layer1, layer2, layer3, layer4} do
        for each bottleneck block b in stage do
            identify:
                conv1_b, conv2_b, conv3_b
                next input layer after b

            S1_out <- importance of conv1_b output channels
            S1_in  <- importance of conv2_b input channels
            S1 <- mean(Normalize(S1_out), Normalize(S1_in))

            S2_out <- importance of conv2_b output channels
            S2_in  <- importance of conv3_b input channels
            S2 <- mean(Normalize(S2_out), Normalize(S2_in))

            S3_out <- importance of conv3_b output channels
            S3_in  <- importance of next layer input channels
            S3 <- mean(Normalize(S3_out), Normalize(S3_in))

            K1, P1, tau1 <- Threshold-Based Channel Selection(S1, M)
            K2, P2, tau2 <- Threshold-Based Channel Selection(S2, M)
            K3, P3, tau3 <- Threshold-Based Channel Selection(S3, M)

            if b uses identity residual connection then
                constrain K3 to be compatible with previous_output_indices
            end if

            prune channels according to:
                K1 for conv1_b output and conv2_b input
                K2 for conv2_b output and conv3_b input
                K3 for conv3_b output and next layer input

            if input and output channel sets are different then
                insert or rebuild residual projection
            end if

            save into full_prune_plan:
                block name
                K1, P1, tau1
                K2, P2, tau2
                K3, P3, tau3
                kept internal channels
                kept output channels

            previous_output_indices <- K3
        end for
    end for

    return full_prune_plan
```

Paper-level description:

```text
Full-Block pruning compresses both the internal bottleneck width and the block
output channels. Since the block output is pruned, residual projections and
subsequent decoder shapes must be adjusted to maintain valid tensor dimensions.
```

## Short Comparison

| Method | Main target | Residual/shape impact | Compression strength |
|---|---|---|---|
| Blueprint | Stage output channels | Medium | Medium |
| Middle Conv2 | Bottleneck internal `conv2` width | Low | Medium |
| Full-Block | Internal and output channels of bottleneck | High | High |
