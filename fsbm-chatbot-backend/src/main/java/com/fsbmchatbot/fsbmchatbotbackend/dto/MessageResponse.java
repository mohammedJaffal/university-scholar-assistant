package com.fsbmchatbot.fsbmchatbotbackend.dto;
import java.time.LocalDateTime;

import com.fsbmchatbot.fsbmchatbotbackend.model.Role;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class MessageResponse {
    private String message;
    private Boolean statue;
    private String type;
     private UserPublicData userPublicData;
    public MessageResponse(String message) {
        this.message = message;
    }
    public MessageResponse(String message,Boolean statue,String type) {
        this.message = message;
        this.statue = statue;
        this.type = type;
    }
    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    public static class UserPublicData {
        private String nom;
        private String prenom;
        private String email;
        private Role role;
        private LocalDateTime verificationCodeExpiry;
    }
   
}