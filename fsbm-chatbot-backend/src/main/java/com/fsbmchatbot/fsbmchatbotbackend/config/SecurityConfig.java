package com.fsbmchatbot.fsbmchatbotbackend.config;

import com.fsbmchatbot.fsbmchatbotbackend.security.CustomUserDetailsService;
import com.fsbmchatbot.fsbmchatbotbackend.security.JwtAuthenticationFilter;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpMethod;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.AuthenticationProvider;
import org.springframework.security.authentication.dao.DaoAuthenticationProvider;
import org.springframework.security.config.annotation.authentication.configuration.AuthenticationConfiguration;
import org.springframework.security.config.annotation.method.configuration.EnableMethodSecurity;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configurers.AbstractHttpConfigurer;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.CorsConfigurationSource;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;

import java.util.Arrays;

@Configuration
@EnableWebSecurity
@EnableMethodSecurity
public class SecurityConfig {

    private final CustomUserDetailsService customUserDetailsService;
    private final JwtAuthenticationFilter jwtAuthenticationFilter;
    private final AuthEntryPointJwt unauthorizedHandler;

    public SecurityConfig(CustomUserDetailsService customUserDetailsService,
                          JwtAuthenticationFilter jwtAuthenticationFilter,
                          AuthEntryPointJwt unauthorizedHandler) {
        this.customUserDetailsService = customUserDetailsService;
        this.jwtAuthenticationFilter = jwtAuthenticationFilter;
        this.unauthorizedHandler = unauthorizedHandler;
    }

    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }

    @Bean
    public AuthenticationProvider authenticationProvider() {
        DaoAuthenticationProvider authProvider = new DaoAuthenticationProvider();
        authProvider.setUserDetailsService(customUserDetailsService);
        authProvider.setPasswordEncoder(passwordEncoder());
        return authProvider;
    }

    @Bean
    public AuthenticationManager authenticationManager(AuthenticationConfiguration config) throws Exception {
        return config.getAuthenticationManager();
    }

    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        http
            .csrf(AbstractHttpConfigurer::disable)
            .cors(cors -> cors.configurationSource(corsConfigurationSource()))
            .exceptionHandling(e -> e.authenticationEntryPoint(unauthorizedHandler))
            .sessionManagement(session -> session.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
            .authorizeHttpRequests(auth -> auth
                // Seul ce endpoint est accessible sans authentification
                .requestMatchers("/api/auth/register/student").permitAll()
                .requestMatchers("/api/auth/verify-email").permitAll()
                .requestMatchers("/api/modules/public").permitAll()
                .requestMatchers("/api/auth/resend-verification-code").permitAll()
                .requestMatchers("/api/auth/login").permitAll()
                .requestMatchers("/api/auth/register/professor").permitAll()
                // Le reste de /api/auth/** nécessite authentification
                .requestMatchers("/api/auth/**").authenticated()

                .requestMatchers(HttpMethod.GET, "/api/filieres/public").permitAll()
                .requestMatchers(HttpMethod.GET, "/api/modules/public").permitAll()
                .requestMatchers("/v3/api-docs/**", "/swagger-ui/**", "/swagger-ui.html").permitAll()

                // Chat access for all roles
                .requestMatchers("/api/chat/**").hasAnyRole("STUDENT", "PROFESSOR", "ADMINISTRATOR")

                // Professors
                .requestMatchers("/api/documents/my-documents").hasRole("PROFESSOR")
                .requestMatchers(HttpMethod.POST, "/api/documents").hasRole("PROFESSOR")
                .requestMatchers(HttpMethod.PUT, "/api/documents/**").hasRole("PROFESSOR")
                .requestMatchers(HttpMethod.DELETE, "/api/documents/**").hasRole("PROFESSOR")
                .requestMatchers(HttpMethod.GET, "/api/documents/download/**").authenticated()
                .requestMatchers(HttpMethod.GET, "/api/modules").hasAnyRole("PROFESSOR", "ADMINISTRATOR")

                // Admins
                .requestMatchers("/api/admin/**").hasRole("ADMINISTRATOR")
                .requestMatchers(HttpMethod.POST, "/api/filieres").hasRole("ADMINISTRATOR")
                .requestMatchers(HttpMethod.PUT, "/api/filieres/**").hasRole("ADMINISTRATOR")
                .requestMatchers(HttpMethod.DELETE, "/api/filieres/**").hasRole("ADMINISTRATOR")
                .requestMatchers(HttpMethod.POST, "/api/modules").hasRole("ADMINISTRATOR")
                .requestMatchers(HttpMethod.PUT, "/api/modules/**").hasRole("ADMINISTRATOR")
                .requestMatchers(HttpMethod.DELETE, "/api/modules/**").hasRole("ADMINISTRATOR")

                // Tout le reste requiert authentification
                .anyRequest().authenticated()
            );

        http.authenticationProvider(authenticationProvider());
        http.addFilterBefore(jwtAuthenticationFilter, UsernamePasswordAuthenticationFilter.class);

        return http.build();
    }

    @Bean
    public CorsConfigurationSource corsConfigurationSource() {
        CorsConfiguration config = new CorsConfiguration();

        // Autorise toutes les origines (pratique en dev, à restreindre en prod)
        config.addAllowedOriginPattern("*");

        config.setAllowedMethods(Arrays.asList("GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"));
        config.setAllowedHeaders(Arrays.asList("Authorization", "Cache-Control", "Content-Type", "X-Requested-With", "Accept"));
        config.setAllowCredentials(true);
        config.setMaxAge(3600L);

        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/**", config);
        return source;
    }
}
