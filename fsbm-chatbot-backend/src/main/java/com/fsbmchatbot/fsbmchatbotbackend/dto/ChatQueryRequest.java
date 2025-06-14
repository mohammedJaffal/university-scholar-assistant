package com.fsbmchatbot.fsbmchatbotbackend.dto;

import lombok.Data;
import lombok.NoArgsConstructor;
@Data
@NoArgsConstructor
public class ChatQueryRequest {
    private String query;
    private Long userId; // Uncomment and use if your Python API needs it
    // Add any other fields your Python API expects in the request
}