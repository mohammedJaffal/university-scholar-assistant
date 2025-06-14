package com.fsbmchatbot.fsbmchatbotbackend.service;

import com.fsbmchatbot.fsbmchatbotbackend.dto.ModuleDTO;
import com.fsbmchatbot.fsbmchatbotbackend.exception.BadRequestException;
import com.fsbmchatbot.fsbmchatbotbackend.exception.ResourceNotFoundException;
import com.fsbmchatbot.fsbmchatbotbackend.model.Module;
import com.fsbmchatbot.fsbmchatbotbackend.repository.ModuleRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Collectors;

@Service
public class ModuleService {

    private static final Logger logger = LoggerFactory.getLogger(ModuleService.class);

    @Autowired
    private ModuleRepository moduleRepository;

    private ModuleDTO convertToModuleDTO(Module module) {
        return new ModuleDTO(
                module.getId(),
                module.getNom(),
                module.getDescription(),
                module.getCode_module()
        );
    }

    @Transactional
    public ModuleDTO createModule(ModuleDTO moduleDTO) {
        if (moduleRepository.existsByNom(moduleDTO.getNom())) {
            throw new BadRequestException("Module with name '" + moduleDTO.getNom() + "' already exists.");
        }
        Module module = new Module();
        module.setNom(moduleDTO.getNom());
        module.setDescription(moduleDTO.getDescription());
        Module savedModule = moduleRepository.save(module);
        logger.info("Created new module: {}", savedModule.getNom());
        return convertToModuleDTO(savedModule);
    }

    @Transactional(readOnly = true)
    public List<ModuleDTO> getAllModules() {
        logger.info("Fetching all modules");
        return moduleRepository.findAll().stream()
                .map(this::convertToModuleDTO)
                .collect(Collectors.toList());
    }

    @Transactional(readOnly = true)
    public ModuleDTO getModuleById(Long moduleId) {
        logger.info("Fetching module with ID: {}", moduleId);
        Module module = moduleRepository.findById(moduleId)
                .orElseThrow(() -> new ResourceNotFoundException("Module", "id", moduleId));
        return convertToModuleDTO(module);
    }

    public Module findModuleEntityById(Long moduleId) { // Used by DocumentService
        return moduleRepository.findById(moduleId)
                .orElseThrow(() -> new ResourceNotFoundException("Module", "id", moduleId));
    }

    @Transactional
    public ModuleDTO updateModule(Long moduleId, ModuleDTO moduleDTO) {
        logger.info("Updating module with ID: {}", moduleId);
        Module module = moduleRepository.findById(moduleId)
                .orElseThrow(() -> new ResourceNotFoundException("Module", "id", moduleId));

        if (!module.getNom().equals(moduleDTO.getNom()) && moduleRepository.existsByNom(moduleDTO.getNom())) {
            throw new BadRequestException("Another module with name '" + moduleDTO.getNom() + "' already exists.");
        }

        module.setNom(moduleDTO.getNom());
        module.setDescription(moduleDTO.getDescription());
        Module updatedModule = moduleRepository.save(module);
        logger.info("Module {} updated successfully", updatedModule.getNom());
        return convertToModuleDTO(updatedModule);
    }

    @Transactional
    public void deleteModule(Long moduleId) {
        logger.info("Deleting module with ID: {}", moduleId);
        Module module = moduleRepository.findById(moduleId)
                .orElseThrow(() -> new ResourceNotFoundException("Module", "id", moduleId));

        if (!module.getDocuments().isEmpty()) {
            throw new BadRequestException("Cannot delete module '" + module.getNom() + "' as it has associated documents.");
        }
        moduleRepository.delete(module);
        logger.info("Module with ID {} deleted successfully", moduleId);
    }
}