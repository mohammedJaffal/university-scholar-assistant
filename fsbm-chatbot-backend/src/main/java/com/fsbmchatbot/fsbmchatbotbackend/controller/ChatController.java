package com.fsbmchatbot.fsbmchatbotbackend.controller;

import com.fsbmchatbot.fsbmchatbotbackend.dto.ChatRequest;
import com.fsbmchatbot.fsbmchatbotbackend.dto.ChatResponse;
import com.fsbmchatbot.fsbmchatbotbackend.security.UserDetailsImpl; // Ensure this import
import com.fsbmchatbot.fsbmchatbotbackend.service.ChatService;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

@CrossOrigin(origins = "*", maxAge = 3600)
@RestController
@RequestMapping("/api/chat")
@PreAuthorize("isAuthenticated()")
public class ChatController {

    @Autowired
    private ChatService chatService;

    @PostMapping("/query")
    public ResponseEntity<ChatResponse> queryChatbot(
            @Valid @RequestBody ChatRequest chatRequest,
            @AuthenticationPrincipal UserDetailsImpl currentUser) {

        Long userId = (currentUser != null) ? currentUser.getId() : null;
        ChatResponse chatResponse = chatService.getQueryResponse(chatRequest.getQuery(), userId);
        return ResponseEntity.ok(chatResponse);
    }

    // Optional: Endpoint for chat history
    // @GetMapping("/history")
    // public ResponseEntity<List<SomeChatHistoryDTO>> getChatHistory(@AuthenticationPrincipal UserDetailsImpl currentUser) {
    //     List<SomeChatHistoryDTO> history = chatService.getUserChatHistory(currentUser.getId());
    //     return ResponseEntity.ok(history);
    // }
}