package com.fsbmchatbot.fsbmchatbotbackend;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.batch.BatchAutoConfiguration;
import org.springframework.cache.annotation.EnableCaching;

@SpringBootApplication(exclude = { BatchAutoConfiguration.class })
@EnableCaching
public class FsbmChatbotBackendApplication {
    public static void main(String[] args) {
        SpringApplication.run(FsbmChatbotBackendApplication.class, args);
    }
}
