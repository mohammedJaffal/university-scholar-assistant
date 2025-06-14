package com.fsbmchatbot.fsbmchatbotbackend.dto;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;
@Data
public class ChatRequest {
    @NotBlank(message = "Query cannot be blank")
    private String query;
}