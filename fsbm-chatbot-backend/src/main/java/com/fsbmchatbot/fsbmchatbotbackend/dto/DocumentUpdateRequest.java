package com.fsbmchatbot.fsbmchatbotbackend.dto;

import jakarta.validation.constraints.NotNull;
import lombok.Data;

@Data
public class DocumentUpdateRequest {
    @NotNull(message = "Module ID cannot be null")
    private Long moduleId; // New Module ID
    // If you allow replacing the file during update, you'd also have MultipartFile here or a separate endpoint
}