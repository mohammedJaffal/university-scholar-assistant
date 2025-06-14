package com.fsbmchatbot.fsbmchatbotbackend.dto;

import jakarta.validation.constraints.NotNull;
import lombok.Data;
// import org.springframework.web.multipart.MultipartFile; // This will be a parameter in controller, not DTO usually

@Data
public class DocumentUploadRequest {
    // MultipartFile will be handled directly in the controller method signature
    // @NotNull(message = "File cannot be null")
    // private MultipartFile file; // File itself

    @NotNull(message = "Module ID cannot be null")
    private Long moduleId; // ID of the module to associate with
}