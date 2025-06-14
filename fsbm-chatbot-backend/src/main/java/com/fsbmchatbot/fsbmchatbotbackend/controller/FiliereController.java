package com.fsbmchatbot.fsbmchatbotbackend.controller;

import com.fsbmchatbot.fsbmchatbotbackend.dto.FiliereDTO;
import com.fsbmchatbot.fsbmchatbotbackend.service.FiliereService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@CrossOrigin(origins = "*", maxAge = 3600)
@RestController
@RequestMapping("/api/filieres")
public class FiliereController {

    @Autowired
    private FiliereService filiereService;

    // Public endpoint to list filieres for forms (e.g., student registration)
    @GetMapping("/public")
    @PreAuthorize("permitAll()") // As per SecurityConfig
    public ResponseEntity<List<FiliereDTO>> getPublicFilieres() {
        List<FiliereDTO> filieres = filiereService.getAllFilieres();
        return ResponseEntity.ok(filieres);
    }

    @GetMapping("/{filiereId}/public")
    @PreAuthorize("permitAll()")
    public ResponseEntity<FiliereDTO> getPublicFiliereById(@PathVariable Long filiereId) {
        FiliereDTO filiere = filiereService.getFiliereById(filiereId);
        return ResponseEntity.ok(filiere);
    }

    // Admin CRUD operations for filieres are in AdminController
}