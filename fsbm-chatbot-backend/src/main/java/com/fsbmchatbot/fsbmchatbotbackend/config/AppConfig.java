package com.fsbmchatbot.fsbmchatbotbackend.config;

import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestTemplate;

import java.time.Duration;

@Configuration
public class AppConfig {

    @Bean
    public RestTemplate restTemplate(RestTemplateBuilder builder) {
        // You can customize the RestTemplate here, e.g., set timeouts
        return builder
                .setConnectTimeout(Duration.ofSeconds(5)) // Connection timeout
                .setReadTimeout(Duration.ofSeconds(30))   // Read timeout (RAG might take time)
                .build();
    }
}