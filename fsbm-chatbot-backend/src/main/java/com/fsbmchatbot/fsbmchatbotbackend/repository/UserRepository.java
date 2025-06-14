package com.fsbmchatbot.fsbmchatbotbackend.repository;

import com.fsbmchatbot.fsbmchatbotbackend.model.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
@Repository
public interface UserRepository extends JpaRepository<User, Long> {
    User findByEmail(String email);
    Boolean existsByEmail(String email);
    Boolean existsByEmailUniversitaire(String emailUniversitaire);
    User findByVerificationCode(String verificationCode);
}