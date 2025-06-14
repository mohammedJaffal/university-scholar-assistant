package com.fsbmchatbot.fsbmchatbotbackend.controller;

import com.fsbmchatbot.fsbmchatbotbackend.dto.ModuleDTO;
import com.fsbmchatbot.fsbmchatbotbackend.service.ModuleService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@CrossOrigin(origins = "*", maxAge = 3600)
@RestController
@RequestMapping("/api/modules")
public class ModuleController {

    @Autowired
    private ModuleService moduleService;

    // Public endpoint to list modules for forms (e.g., student registration, prof document upload)
    @PostMapping("/public")
    @PreAuthorize("permitAll()") // As per SecurityConfig
    public ResponseEntity<List<ModuleDTO>> getPublicModules() {
        List<ModuleDTO> modules = moduleService.getAllModules();
        return ResponseEntity.ok(modules);
    }

    // Optional: If you need an authenticated endpoint to get modules, distinct from public
    // This might be useful if different details are shown or different filtering applies
    @GetMapping
    @PreAuthorize("hasAnyRole('PROFESSOR', 'ADMINISTRATOR')") // Example: for internal forms
    public ResponseEntity<List<ModuleDTO>> getAuthenticatedModules() {
        List<ModuleDTO> modules = moduleService.getAllModules(); // Or a different service method
        return ResponseEntity.ok(modules);
    }


    @GetMapping("/{moduleId}/public")
    @PreAuthorize("permitAll()")
    public ResponseEntity<ModuleDTO> getPublicModuleById(@PathVariable Long moduleId) {
        ModuleDTO module = moduleService.getModuleById(moduleId);
        return ResponseEntity.ok(module);
    }

    // Admin CRUD operations for modules are in AdminController
}