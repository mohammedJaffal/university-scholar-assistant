package com.fsbmchatbot.fsbmchatbotbackend.service;

import com.fsbmchatbot.fsbmchatbotbackend.dto.ChatQueryRequest; // For Python API request
import com.fsbmchatbot.fsbmchatbotbackend.dto.ChatQueryResponse; // For Python API response
import com.fsbmchatbot.fsbmchatbotbackend.dto.ChatResponse; // For Spring Controller
import com.fsbmchatbot.fsbmchatbotbackend.exception.AppException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.HttpServerErrorException;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestTemplate;

import java.util.Collections;

@Service
public class ChatService {

    private static final Logger logger = LoggerFactory.getLogger(ChatService.class);

    private final RestTemplate restTemplate;

    @Value("${python.chatbot.api.url}")
    private String pythonChatbotApiUrl;

    @Autowired
    public ChatService(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    public ChatResponse getQueryResponse(String userQuery, Long... optionalUserId) {
        logger.info("Sending query to Python RAG API: '{}'", userQuery);
        Long userId = (optionalUserId != null && optionalUserId.length > 0) ? optionalUserId[0] : null;
        if (userId != null) {
            logger.info("User context ID for query: {}", userId);
        }

        ChatQueryRequest pythonApiRequest = new ChatQueryRequest();
        pythonApiRequest.setQuery(userQuery);
        // if (userId != null) {
        // pythonApiRequest.setUserId(userId); // If your Python API DTO has userId
        // }

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        headers.setAccept(Collections.singletonList(MediaType.APPLICATION_JSON));

        HttpEntity<ChatQueryRequest> entity = new HttpEntity<>(pythonApiRequest, headers);

        try {
            ResponseEntity<ChatQueryResponse> responseEntity = restTemplate.exchange(
                    pythonChatbotApiUrl,
                    HttpMethod.POST,
                    entity,
                    ChatQueryResponse.class
            );

            if (responseEntity.getStatusCode() == HttpStatus.OK && responseEntity.getBody() != null) {
                ChatQueryResponse pythonApiResponse = responseEntity.getBody();
                logger.info("Received response from Python RAG API: '{}'", pythonApiResponse.getResponse());
                return new ChatResponse(pythonApiResponse.getResponse()); // Adapt to Spring DTO
            } else {
                logger.error("Python RAG API returned non-OK status: {} or empty body", responseEntity.getStatusCode());
                throw new AppException(HttpStatus.INTERNAL_SERVER_ERROR, "Failed to get a valid response from the chatbot service.");
            }

        } catch (HttpClientErrorException | HttpServerErrorException e) {
            logger.error("Error response from Python RAG API (Status {}): {}", e.getStatusCode(), e.getResponseBodyAsString(), e);
            throw new AppException(HttpStatus.BAD_GATEWAY, "Chatbot service returned an error: " + e.getStatusCode());
        } catch (ResourceAccessException e) {
            logger.error("Cannot access Python RAG API at {}: {}", pythonChatbotApiUrl, e.getMessage(), e);
            throw new AppException(HttpStatus.SERVICE_UNAVAILABLE, "Chatbot service is currently unavailable. Please try again later.");
        } catch (Exception e) {
            logger.error("Unexpected error while calling Python RAG API: {}", e.getMessage(), e);
            throw new AppException(HttpStatus.INTERNAL_SERVER_ERROR, "An unexpected error occurred while communicating with the chatbot service.");
        }
    }
}