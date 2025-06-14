package com.fsbmchatbot.fsbmchatbotbackend.service;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

import lombok.Builder;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.fsbmchatbot.fsbmchatbotbackend.dto.AuthResponse;
import com.fsbmchatbot.fsbmchatbotbackend.dto.LoginRequest;
import com.fsbmchatbot.fsbmchatbotbackend.dto.MessageResponse;
import com.fsbmchatbot.fsbmchatbotbackend.dto.ProfessorRegisterRequest;
import com.fsbmchatbot.fsbmchatbotbackend.dto.StudentRegisterRequest;
import com.fsbmchatbot.fsbmchatbotbackend.dto.VerifyEmailRequest;
import com.fsbmchatbot.fsbmchatbotbackend.exception.ResourceNotFoundException;
import com.fsbmchatbot.fsbmchatbotbackend.model.Role;
import com.fsbmchatbot.fsbmchatbotbackend.model.User;
import com.fsbmchatbot.fsbmchatbotbackend.model.Module;
import com.fsbmchatbot.fsbmchatbotbackend.repository.FiliereRepository;
import com.fsbmchatbot.fsbmchatbotbackend.repository.ModuleRepository;
import com.fsbmchatbot.fsbmchatbotbackend.repository.UserRepository;
import com.fsbmchatbot.fsbmchatbotbackend.security.JwtService;
import com.fsbmchatbot.fsbmchatbotbackend.security.UserDetailsImpl;

@Service
public class AuthService {
    private static final Logger logger = LoggerFactory.getLogger(AuthService.class);
    private final AuthenticationManager authenticationManager;
    private final UserRepository userRepository;
    private final ModuleRepository moduleRepository;
    private final PasswordEncoder passwordEncoder;
    private final JwtService jwtService;
    private static final int VERIFICATION_CODE_EXPIRY_MINUTES = 15;
    @Autowired
    private EmailService emailService;
    public AuthService(UserRepository userRepository, AuthenticationManager authenticationManager, FiliereRepository filiereRepository, PasswordEncoder passwordEncoder, JwtService jwtService,ModuleRepository moduleRepository) {
        this.userRepository = userRepository;
        this.authenticationManager = authenticationManager;
        this.passwordEncoder = passwordEncoder;
        this.jwtService = jwtService;
        this.moduleRepository=moduleRepository;
    }

    @Builder
   @Transactional
public ArrayList<MessageResponse> registerStudent(StudentRegisterRequest registerRequest) {
    ArrayList<MessageResponse> ListMsg = new ArrayList<>();

    if (!registerRequest.getPassword().equals(registerRequest.getConfirmPassword())) {
        ListMsg.add(new MessageResponse("Passwords do not match.", false, "password"));
    }

    if (userRepository.existsByEmail(registerRequest.getEmail())) {
        ListMsg.add(new MessageResponse("Error: Email is already in use!", false, "email"));
    }

    if (registerRequest.getEmailUniversitaire() != null && !registerRequest.getEmailUniversitaire().isEmpty()
            && userRepository.existsByEmailUniversitaire(registerRequest.getEmailUniversitaire())) {
        ListMsg.add(new MessageResponse("Error: University Email is already in use!", false, "email-universitaire"));
    }

    if (!ListMsg.isEmpty()) {
        return ListMsg;
    }

    // Création utilisateur
    User user = new User();
    user.setEmail(registerRequest.getEmail());
    user.setEmailUniversitaire(registerRequest.getEmailUniversitaire());
    user.setNom(registerRequest.getNom());
    user.setPrenom(registerRequest.getPrenom());
    user.setRole(Role.ROLE_STUDENT);
    user.setFiliere(registerRequest.getFiliereNom());
    user.setPassword(passwordEncoder.encode(registerRequest.getPassword()));
    user.setActive(false);
    user.setVerified(false);

    // Génération du code AVANT le save
    String verificationCode = generateVerificationCode();
    user.setVerificationCode(verificationCode);
    user.setVerificationCodeExpiry(LocalDateTime.now().plusMinutes(VERIFICATION_CODE_EXPIRY_MINUTES));

    userRepository.save(user);

    logger.info("Student registered successfully: {}", user.getEmail());
    logger.info("SIMULATION: Verification email sent to {} with code: {}", user.getEmail(), verificationCode);

    // Envoi de l’e-mail de vérification
    emailService.sendSimpleEmail(user.getEmail(), "Verification Code", verificationCode);

    // Construction du message de réponse
    LocalDateTime now = LocalDateTime.now();
    MessageResponse.UserPublicData userData = new MessageResponse.UserPublicData(
        registerRequest.getNom(), registerRequest.getPrenom(), registerRequest.getEmail(), Role.ROLE_STUDENT, now
    );

    MessageResponse msg = new MessageResponse(
        "Compte créé (simulation). Veuillez vérifier votre e-mail.", true, "CreateAccount"
    );
    msg.setUserPublicData(userData);

    ListMsg.add(msg);
    return ListMsg;
}

    @Transactional
public ArrayList<MessageResponse> registerProfessor(ProfessorRegisterRequest registerRequest) {
    ArrayList<MessageResponse> ListMsg = new ArrayList<>();
    boolean isValid = true;

    if (!registerRequest.getPassword().equals(registerRequest.getConfirmPassword())) {
        isValid = false;
        ListMsg.add(new MessageResponse("Passwords do not match.", false, "password"));
    }

    if (userRepository.existsByEmail(registerRequest.getEmail())) {
        isValid = false;
        ListMsg.add(new MessageResponse("Error: Email is already in use!", false, "email"));
    }

    if (!isValid) {
        return ListMsg;
    }

    String verificationCode = generateVerificationCode();
    User user = new User(
            registerRequest.getNom(),
            registerRequest.getPrenom(),
            registerRequest.getEmail(),
            passwordEncoder.encode(registerRequest.getPassword()),
            Role.ROLE_PROFESSOR
    );

    user.setActive(false);
    user.setVerified(false);
    user.setVerificationCode(verificationCode);
    user.setVerificationCodeExpiry(LocalDateTime.now().plusMinutes(VERIFICATION_CODE_EXPIRY_MINUTES));

    userRepository.save(user);
    logger.info("Professor registered successfully: {}", user.getEmail());

    LocalDateTime now = LocalDateTime.now();
    MessageResponse.UserPublicData userData = new MessageResponse.UserPublicData(
            registerRequest.getNom(),
            registerRequest.getPrenom(),
            registerRequest.getEmail(),
            Role.ROLE_PROFESSOR,
            now
    );

    MessageResponse msg = new MessageResponse(
            "Compte créé (simulation). Veuillez vérifier votre e-mail.",
            true,
            "CreateAccount"
    );
    msg.setUserPublicData(userData);
    ListMsg.add(msg);
    // Simuler l’envoi d’email
    emailService.sendSimpleEmail(registerRequest.getEmail(), "Verification Code", verificationCode);
    logger.info("SIMULATION: Verification email sent to {} with code: {}", user.getEmail(), verificationCode);
    return ListMsg;
}

    public List<Module> GetAllModule(){
        List<Module> modules=new ArrayList<>();
        modules =moduleRepository.findAll();
        return modules;
    }
  public AuthResponse loginUser(LoginRequest loginRequest) {
    System.out.println("Etape 1");

    Authentication authentication = authenticationManager.authenticate(
        new UsernamePasswordAuthenticationToken(
            loginRequest.getEmail(), 
            loginRequest.getPassword()
        )
    );

    System.out.println("Etape 2");

    SecurityContextHolder.getContext().setAuthentication(authentication);
    String jwt = jwtService.generateToken(authentication);

    System.out.println("Etape 3");

    UserDetailsImpl userDetails = (UserDetailsImpl) authentication.getPrincipal();
    User user = userRepository.findById(userDetails.getId())
        .orElseThrow(() -> new ResourceNotFoundException("User", "id", userDetails.getId()));

    System.out.println("Etape 4");

    if (!user.isVerified()) {
        logger.warn("User {} logged in but account is not verified.", user.getEmail());
    }

    System.out.println("Etape 5");

    if (!user.isActive()) {
        logger.warn("User {} logged in but account is not active.", user.getEmail());
    }

    System.out.println("Etape 6");

    logger.info("User {} logged in successfully.", userDetails.getUsername());

    return new AuthResponse(
        jwt,
        userDetails.getId(),
        userDetails.getUsername(), // email
        user.getNom(),
        user.getPrenom(),
        user.getRole(),
        user.getFiliere() != null ? user.getFiliere() : null,
        user.isActive(),
        user.isVerified()
    );
}

   @Transactional
public MessageResponse verifyEmail(VerifyEmailRequest verifyEmailRequest) {
    User user;
    try {
        user = userRepository.findByVerificationCode(verifyEmailRequest.getVerificationCode());
        if (user == null) {
            return new MessageResponse("The verification code is incorrect.", false, "verify-email");
        }
    } catch (Exception e) {
        return new MessageResponse("An error occurred while verifying the code.", false, "verify-email");
    }

    if (user.isVerified()) {
        return new MessageResponse("Account already verified.", true, "verify-email");
    }

    if (user.getVerificationCodeExpiry() != null && user.getVerificationCodeExpiry().isBefore(LocalDateTime.now())) {
        return new MessageResponse("Verification code has expired. Please request a new one.", false, "verify-email");
    }

    // Vérification réussie
    user.setVerified(true);
    user.setActive(true);
    user.setVerificationCode(null);
    user.setVerificationCodeExpiry(null);
    userRepository.save(user);

    logger.info("Email verified successfully for user: {}", user.getEmail());

    return new MessageResponse("Email verified successfully. Your account will be reviewed by an administrator for activation.", true, "verify-email");
}

    @Transactional
    public MessageResponse resendVerificationCode(VerifyEmailRequest verifyEmailRequest ) {
        User user = new User();
        try {
            user=userRepository.findByEmail(verifyEmailRequest.getEmail());
            if(user == null){
                return new MessageResponse("This email was not found.",false,"verify-email");
            }
        } catch (Exception e) {
             return new MessageResponse("This email was not found.",false,"verify-email");
        }
        if (user.isVerified()) {
             return new MessageResponse("Account is already verified.",true,"verify-email");
        }
        System.out.println("test 1");
        String newVerificationCode = generateVerificationCode();
        user.setVerificationCode(newVerificationCode);
        user.setVerificationCodeExpiry(LocalDateTime.now().plusMinutes(VERIFICATION_CODE_EXPIRY_MINUTES));
        userRepository.save(user);
        emailService.sendSimpleEmail(user.getEmail(), "Verification Code",newVerificationCode);
        logger.info("SIMULATION: New verification email sent to {} with code: {}", user.getEmail(), newVerificationCode);
        return new MessageResponse("Nouveau code de vérification envoyé (simulation). Code: ",true,"resend-verify-email");
    }
    private String generateVerificationCode() {
        // Generate a simple 6-digit code for simulation
        // For production, use a more secure, longer, random string (e.g., UUID)
        return String.format("%06d", (int) (Math.random() * 1000000));
        // return UUID.randomUUID().toString().substring(0, 8).toUpperCase(); // Alternative
    }


    // Optional: Method to register an Admin (perhaps via CommandLineRunner or a secured endpoint)
    @Builder
    @Transactional
    public void registerAdminIfNotExists(String email, String password, String nom, String prenom) {
        if (!userRepository.existsByEmail(email)) {
            User adminUser = new User(
                    nom,
                    prenom,
                    email,
                    passwordEncoder.encode(password),
                    Role.ROLE_ADMINISTRATOR
            );

            adminUser.setActive(true);
            adminUser.setVerified(true);

            userRepository.save(adminUser);
            logger.info("Admin user {} created.", email);
        } else {
            logger.info("Admin user {} already exists.", email);
        }
    }
}