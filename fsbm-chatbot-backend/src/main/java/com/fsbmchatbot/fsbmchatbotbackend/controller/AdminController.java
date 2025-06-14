package com.fsbmchatbot.fsbmchatbotbackend.controller;

import com.fsbmchatbot.fsbmchatbotbackend.dto.FiliereDTO;
import com.fsbmchatbot.fsbmchatbotbackend.dto.MessageResponse;
import com.fsbmchatbot.fsbmchatbotbackend.dto.ModuleDTO;
import com.fsbmchatbot.fsbmchatbotbackend.dto.UserDTO;
import com.fsbmchatbot.fsbmchatbotbackend.service.FiliereService;
import com.fsbmchatbot.fsbmchatbotbackend.service.ModuleService;
import com.fsbmchatbot.fsbmchatbotbackend.service.UserService;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@CrossOrigin(origins = "*", maxAge = 3600)
@RestController
@RequestMapping("/api/admin")
@PreAuthorize("hasRole('ADMINISTRATOR')")
public class AdminController {

    @Autowired
    private UserService userService;

    @Autowired
    private FiliereService filiereService;

    @Autowired
    private ModuleService moduleService;

    // --- User Management ---
    @GetMapping("/users")
    public ResponseEntity<List<UserDTO>> getAllUsers() {
        List<UserDTO> users = userService.getAllUsers();
        return ResponseEntity.ok(users);
    }

    @GetMapping("/users/{userId}")
    public ResponseEntity<UserDTO> getUserById(@PathVariable Long userId) {
        UserDTO user = userService.getUserById(userId);
        return ResponseEntity.ok(user);
    }

    @PatchMapping("/users/{userId}/activate")
    public ResponseEntity<MessageResponse> activateUser(@PathVariable Long userId) {
        userService.activateUser(userId);
        return ResponseEntity.ok(new MessageResponse("User account activated successfully."));
    }

    @PatchMapping("/users/{userId}/deactivate")
    public ResponseEntity<MessageResponse> deactivateUser(@PathVariable Long userId) {
        userService.deactivateUser(userId);
        return ResponseEntity.ok(new MessageResponse("User account deactivated successfully."));
    }

    // --- Filiere Management ---
    @PostMapping("/filieres")
    public ResponseEntity<FiliereDTO> createFiliere(@Valid @RequestBody FiliereDTO filiereDTO) {
        FiliereDTO createdFiliere = filiereService.createFiliere(filiereDTO);
        return new ResponseEntity<>(createdFiliere, HttpStatus.CREATED);
    }

    @GetMapping("/filieres") // Admin view of all filieres
    public ResponseEntity<List<FiliereDTO>> getAllFilieresAdmin() {
        List<FiliereDTO> filieres = filiereService.getAllFilieres();
        return ResponseEntity.ok(filieres);
    }

    @GetMapping("/filieres/{filiereId}")
    public ResponseEntity<FiliereDTO> getFiliereByIdAdmin(@PathVariable Long filiereId) {
        FiliereDTO filiere = filiereService.getFiliereById(filiereId);
        return ResponseEntity.ok(filiere);
    }

    @PutMapping("/filieres/{filiereId}")
    public ResponseEntity<FiliereDTO> updateFiliere(@PathVariable Long filiereId, @Valid @RequestBody FiliereDTO filiereDTO) {
        FiliereDTO updatedFiliere = filiereService.updateFiliere(filiereId, filiereDTO);
        return ResponseEntity.ok(updatedFiliere);
    }

    @DeleteMapping("/filieres/{filiereId}")
    public ResponseEntity<MessageResponse> deleteFiliere(@PathVariable Long filiereId) {
        filiereService.deleteFiliere(filiereId);
        return ResponseEntity.ok(new MessageResponse("Filiere deleted successfully."));
    }

    // --- Module Management ---
    @PostMapping("/modules")
    public ResponseEntity<ModuleDTO> createModule(@Valid @RequestBody ModuleDTO moduleDTO) {
        ModuleDTO createdModule = moduleService.createModule(moduleDTO);
        return new ResponseEntity<>(createdModule, HttpStatus.CREATED);
    }

    @GetMapping("/modules") // Admin view of all modules
    public ResponseEntity<List<ModuleDTO>> getAllModulesAdmin() {
        List<ModuleDTO> modules = moduleService.getAllModules();
        return ResponseEntity.ok(modules);
    }

    @GetMapping("/modules/{moduleId}")
    public ResponseEntity<ModuleDTO> getModuleByIdAdmin(@PathVariable Long moduleId) {
        ModuleDTO module = moduleService.getModuleById(moduleId);
        return ResponseEntity.ok(module);
    }

    @PutMapping("/modules/{moduleId}")
    public ResponseEntity<ModuleDTO> updateModule(@PathVariable Long moduleId, @Valid @RequestBody ModuleDTO moduleDTO) {
        ModuleDTO updatedModule = moduleService.updateModule(moduleId, moduleDTO);
        return ResponseEntity.ok(updatedModule);
    }

    @DeleteMapping("/modules/{moduleId}")
    public ResponseEntity<MessageResponse> deleteModule(@PathVariable Long moduleId) {
        moduleService.deleteModule(moduleId);
        return ResponseEntity.ok(new MessageResponse("Module deleted successfully."));
    }
}