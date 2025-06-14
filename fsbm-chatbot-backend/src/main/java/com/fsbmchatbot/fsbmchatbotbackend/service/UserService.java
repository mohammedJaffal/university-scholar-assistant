package com.fsbmchatbot.fsbmchatbotbackend.service;

import com.fsbmchatbot.fsbmchatbotbackend.dto.UserDTO;
import com.fsbmchatbot.fsbmchatbotbackend.exception.ResourceNotFoundException;
import com.fsbmchatbot.fsbmchatbotbackend.exception.BadRequestException; // Added for self-deactivation check
import com.fsbmchatbot.fsbmchatbotbackend.model.User;
import com.fsbmchatbot.fsbmchatbotbackend.repository.UserRepository;
import com.fsbmchatbot.fsbmchatbotbackend.security.UserDetailsImpl; // If needed for current admin check
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.security.core.context.SecurityContextHolder; // If needed for current admin check
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Collectors;

@Service
public class UserService {

    private static final Logger logger = LoggerFactory.getLogger(UserService.class);

    private final UserRepository userRepository;

    public UserService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    private UserDTO convertToUserDTO(User user) {
        return new UserDTO(
                user.getId(),
                user.getNom(),
                user.getPrenom(),
                user.getEmail(),
                user.getEmailUniversitaire(),
                user.getRole(),
                user.getFiliere() != null ? user.getFiliere() : null,
                user.isActive(),
                user.isVerified(),
                user.getCreatedAt()
        );
    }

    @Transactional(readOnly = true)
    public List<UserDTO> getAllUsers() {
        logger.info("Fetching all users");
        return userRepository.findAll().stream()
                .map(this::convertToUserDTO)
                .collect(Collectors.toList());
    }

    @Transactional(readOnly = true)
    public UserDTO getUserById(Long userId) {
        logger.info("Fetching user with ID: {}", userId);
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new ResourceNotFoundException("User", "id", userId));
        return convertToUserDTO(user);
    }

    @Transactional
    public void activateUser(Long userId) {
        logger.info("Activating user with ID: {}", userId);
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new ResourceNotFoundException("User", "id", userId));

        if (user.isActive()) {
            logger.warn("User {} is already active.", userId);
            return;
        }
        user.setActive(true);
        userRepository.save(user);
        logger.info("User {} activated successfully.", userId);
    }

    @Transactional
    public void deactivateUser(Long userId) {
        logger.info("Deactivating user with ID: {}", userId);

        // Prevent admin from deactivating themselves
        Object principal = SecurityContextHolder.getContext().getAuthentication().getPrincipal();
        if (principal instanceof UserDetailsImpl currentUserDetails) {
            if (currentUserDetails.getId().equals(userId)) {
                logger.warn("Admin user {} attempted to deactivate their own account.", userId);
                throw new BadRequestException("Administrators cannot deactivate their own account.");
            }
        }

        User user = userRepository.findById(userId)
                .orElseThrow(() -> new ResourceNotFoundException("User", "id", userId));

        if (!user.isActive()) {
            logger.warn("User {} is already inactive.", userId);
            return;
        }
        user.setActive(false);
        userRepository.save(user);
        logger.info("User {} deactivated successfully.", userId);
    }

    public User findUserById(Long userId) {
        return userRepository.findById(userId)
                .orElseThrow(() -> new ResourceNotFoundException("User", "id", userId));
    }
}